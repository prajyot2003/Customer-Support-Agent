"""Step 3: registry whitelist + safe dispatcher with argument validation.

The model can only reach functions listed in REGISTRY; everything else is rejected.
"""
import json
import threading

from tools import (TOOLS, check_inventory, create_exchange, lookup_order, process_refund,
                   send_notification)

REGISTRY = {"lookup_order": lookup_order, "check_inventory": check_inventory,
            "create_exchange": create_exchange, "process_refund": process_refund,
            "send_notification": send_notification}
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
# Tool calls from one model turn run on worker threads (agent.py). They share one SQLite connection and the
# tools.py cache, so execution is serialised here; validation above still runs concurrently.
# ponytail: one global lock; use a connection per thread if tools start doing slow network I/O.
_TOOL_LOCK = threading.Lock()


def dispatch(name, raw_args):
    if name not in REGISTRY:
        return {"status": "error", "error": "UNAUTHORIZED_TOOL", "message": f"Tool '{name}' is not allowed."}
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError as e:
        return {"status": "error", "error": "MALFORMED_ARGUMENTS", "message": f"Arguments are not valid JSON: {e}"}
    if not isinstance(args, dict):
        return {"status": "error", "error": "MALFORMED_ARGUMENTS", "message": "Arguments must be a JSON object."}
    schema = SCHEMAS[name]
    missing = [k for k in schema["required"] if k not in args]
    unknown = [k for k in args if k not in schema["properties"]]
    bad_type = [k for k, v in args.items() if k in schema["properties"] and not isinstance(v, str)]
    empty = [k for k, v in args.items() if isinstance(v, str) and not v.strip()]
    if missing or unknown or bad_type or empty:
        return {"status": "error", "error": "INVALID_ARGUMENTS",
                "message": f"missing={missing} unknown={unknown} not_string={bad_type} empty={empty}"}
    try:
        with _TOOL_LOCK:
            return REGISTRY[name](**args)
    except Exception as e:  # a tool bug must never crash the agent loop
        return {"status": "error", "error": "TOOL_FAILURE", "message": str(e)}
