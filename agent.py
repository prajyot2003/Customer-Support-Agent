"""Decision-Making Resolution Agent (Groq tool-calling + ReAct loop).

Run:   python agent.py "My order ORD-1002 arrived damaged"
Needs: pip install groq ; GROQ_API_KEY in .env (or exported in the shell)
"""
import json
import os
import sqlite3
import sys
import uuid


def _load_dotenv(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
    """Read KEY=value lines from .env into the environment (real env vars take priority)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                value = value.strip().strip('"').strip("'")
                if value and not os.environ.get(key.strip()):
                    os.environ[key.strip()] = value


_load_dotenv()
MODEL =os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_ITERATIONS = 6  # lookup, inventory, exchange/refund, notify, final answer + 1 retry

# --------------------------------------------------------------------------
# Database (SQLite file orders.db, seeded on first run; changes persist across runs)
# --------------------------------------------------------------------------
SEED = """
CREATE TABLE orders (order_id TEXT PRIMARY KEY, customer_name TEXT, customer_email TEXT,
    item_id TEXT, amount REAL, status TEXT, exchange_eligible INTEGER, eligibility_note TEXT);
CREATE TABLE inventory (item_id TEXT PRIMARY KEY, item_name TEXT, warehouse TEXT, stock INTEGER);
CREATE TABLE refunds (refund_id TEXT PRIMARY KEY, order_id TEXT UNIQUE, amount REAL, reason TEXT);
CREATE TABLE exchanges (exchange_id TEXT PRIMARY KEY, order_id TEXT UNIQUE, item_id TEXT);
CREATE TABLE notifications (id INTEGER PRIMARY KEY, email TEXT, message TEXT);

INSERT INTO inventory VALUES
 ('SKU-HEADPHONES', 'Wireless Headphones', 'Pune-WH1', 12),
 ('SKU-KETTLE',     'Electric Kettle',     'Pune-WH1', 0),
 ('SKU-LAMP',       'Desk Lamp',           'Mumbai-WH2', 5);
INSERT INTO orders VALUES
 ('ORD-1001', 'Priya Sharma', 'priya@example.com', 'SKU-HEADPHONES', 89.99, 'delivered', 1, 'Within 30-day exchange window'),
 ('ORD-1002', 'Arjun Mehta',  'arjun@example.com', 'SKU-KETTLE',     34.50, 'delivered', 1, 'Within 30-day exchange window'),
 ('ORD-1003', 'Meera Iyer',   'meera@example.com', 'SKU-LAMP',       27.00, 'delivered', 0, 'Outside 30-day exchange window');
"""


DB_PATH = os.environ.get("ORDERS_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "orders.db"))


def init_db(path=None):
    """Open the orders database, creating + seeding it on first use. Delete orders.db to reset."""
    path = path or DB_PATH
    is_new = path == ":memory:" or not os.path.exists(path)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    if is_new:
        db.executescript(SEED)
    return db


DB = init_db()


def _order(order_id):
    return DB.execute("SELECT * FROM orders WHERE order_id = ?", (order_id.strip().upper(),)).fetchone()


def _stock(item_id):
    row = DB.execute("SELECT stock FROM inventory WHERE item_id = ?", (item_id,)).fetchone()
    return row["stock"] if row else 0


# --------------------------------------------------------------------------
# Step 1: Tools — deterministic, always return structured dicts
# --------------------------------------------------------------------------
def lookup_order(order_id):
    o = _order(order_id)
    if not o:
        return {"status": "error", "error": "ORDER_NOT_FOUND",
                "message": f"No order with id '{order_id}'. Ask the customer to verify the order ID."}
    return {"status": "success", "order_id": o["order_id"], "customer_name": o["customer_name"],
            "customer_email": o["customer_email"], "item_id": o["item_id"], "amount": o["amount"],
            "order_status": o["status"], "exchange_eligible": bool(o["exchange_eligible"]),
            "eligibility_note": o["eligibility_note"]}


def check_inventory(item_id):
    row = DB.execute("SELECT * FROM inventory WHERE item_id = ?", (item_id.strip().upper(),)).fetchone()
    if not row:
        return {"status": "error", "error": "ITEM_NOT_FOUND", "message": f"Unknown item id '{item_id}'."}
    return {"status": "success", "item_id": row["item_id"], "item_name": row["item_name"],
            "warehouse": row["warehouse"], "stock": row["stock"], "available": row["stock"] > 0}


def create_exchange(order_id):
    o = _order(order_id)
    if not o:
        return {"status": "error", "error": "ORDER_NOT_FOUND", "message": f"No order '{order_id}'."}
    # Policy enforced in code, not just in the prompt.
    if not o["exchange_eligible"]:
        return {"status": "error", "error": "NOT_ELIGIBLE", "message": "Order is not eligible for exchange; issue a refund instead."}
    if _stock(o["item_id"]) <= 0:
        return {"status": "error", "error": "OUT_OF_STOCK", "message": "No replacement units in stock; issue a refund instead."}
    if DB.execute("SELECT 1 FROM exchanges WHERE order_id = ?", (o["order_id"],)).fetchone():
        return {"status": "error", "error": "ALREADY_EXCHANGED", "message": "An exchange already exists for this order."}
    exchange_id = "EXC-" + uuid.uuid4().hex[:8].upper()
    with DB:
        DB.execute("UPDATE inventory SET stock = stock - 1 WHERE item_id = ?", (o["item_id"],))
        DB.execute("INSERT INTO exchanges VALUES (?, ?, ?)", (exchange_id, o["order_id"], o["item_id"]))
    return {"status": "success", "exchange_id": exchange_id, "order_id": o["order_id"], "item_id": o["item_id"]}


def process_refund(order_id, reason):
    o = _order(order_id)
    if not o:
        return {"status": "error", "error": "ORDER_NOT_FOUND", "message": f"No order '{order_id}'."}
    # Money path guard: refunds only when an exchange is impossible.
    if o["exchange_eligible"] and _stock(o["item_id"]) > 0:
        return {"status": "error", "error": "EXCHANGE_AVAILABLE",
                "message": "Replacement is in stock and order is eligible; create an exchange instead of refunding."}
    if DB.execute("SELECT 1 FROM refunds WHERE order_id = ?", (o["order_id"],)).fetchone():
        return {"status": "error", "error": "ALREADY_REFUNDED", "message": "This order was already refunded."}
    refund_id = "RFD-" + uuid.uuid4().hex[:8].upper()
    with DB:  # amount comes from the DB, never from the model
        DB.execute("INSERT INTO refunds VALUES (?, ?, ?, ?)", (refund_id, o["order_id"], o["amount"], reason))
    return {"status": "success", "refund_id": refund_id, "order_id": o["order_id"],
            "amount_refunded": o["amount"], "method": "original payment method"}


def send_notification(email, message):
    if "@" not in email:
        return {"status": "error", "error": "INVALID_EMAIL", "message": f"'{email}' is not a valid email."}
    with DB:
        cur = DB.execute("INSERT INTO notifications (email, message) VALUES (?, ?)", (email, message))
    return {"status": "success", "notification_id": cur.lastrowid, "sent_to": email}


# --------------------------------------------------------------------------
# Step 2: Declarative tool contracts (what the model sees)
# --------------------------------------------------------------------------
def _schema(name, description, params):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object",
                       "properties": {k: {"type": "string", "description": d} for k, d in params.items()},
                       "required": list(params)}}}


TOOLS = [
    _schema("lookup_order",
            "Look up an order in the database. Returns customer name, email, item_id, amount, order status "
            "and whether the order is eligible for exchange. Always call this first.",
            {"order_id": "Order identifier, e.g. 'ORD-1001'."}),
    _schema("check_inventory",
            "Check real-time warehouse stock for an item to see if a replacement unit is available.",
            {"item_id": "Item identifier from lookup_order, e.g. 'SKU-KETTLE'."}),
    _schema("create_exchange",
            "Confirm a replacement exchange. Use ONLY when the order is exchange-eligible AND stock > 0.",
            {"order_id": "Order identifier to exchange."}),
    _schema("process_refund",
            "Issue a financial refund to the customer's original payment method. Use ONLY when a replacement "
            "is unavailable (stock is 0) or the order is not eligible for exchange.",
            {"order_id": "Order identifier to refund.", "reason": "Short reason for the refund."}),
    _schema("send_notification",
            "Send the customer an official confirmation email summarising the resolution. "
            "Call after an exchange or refund succeeds.",
            {"email": "Customer email from lookup_order.", "message": "Email body text."}),
]

# --------------------------------------------------------------------------
# Step 3: Registry whitelist + safe dispatcher with argument validation
# --------------------------------------------------------------------------
REGISTRY = {"lookup_order": lookup_order, "check_inventory": check_inventory,
            "create_exchange": create_exchange, "process_refund": process_refund,
            "send_notification": send_notification}
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}


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
        return REGISTRY[name](**args)
    except Exception as e:  # a tool bug must never crash the agent loop
        return {"status": "error", "error": "TOOL_FAILURE", "message": str(e)}


# --------------------------------------------------------------------------
# Step 4 + 5: ReAct loop with iteration ceiling and escalation
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an automated order resolution specialist.
When a customer reports a problem with an item:
1. Call lookup_order with their order ID. If it is not found, apologise, say the order was not found,
   and ask them to verify the order ID. Do not call other tools.
2. If the order is exchange-eligible, call check_inventory for its item_id.
3. If eligible AND stock > 0: call create_exchange. Otherwise (stock is 0 or not eligible): call process_refund.
4. After a successful exchange or refund, call send_notification to the customer's email.
5. Finish with a short, polite summary for the customer (include exchange/refund reference and amount).
If no order ID was given, ask for it. Never invent data; only use tool results.
If a tool returns an error, read it and adjust (fix arguments or follow its instruction)."""


def run_agent(user_message, client=None, verbose=True, messages=None):
    """Returns {"reply": str, "escalated": bool, "trace": [tool names called]}.
    Pass the same `messages` list across calls to keep a multi-turn conversation."""
    if client is None:
        from groq import Groq
        client = Groq()
    if messages is None:
        messages = []
    if not messages:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_message})
    trace = []
    for _ in range(MAX_ITERATIONS):
        try:
            resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS,
                                                  tool_choice="auto", temperature=0)
        except Exception as e:
            if "tool_use_failed" not in str(e):  # auth, retired model, rate limit: surface, don't retry
                raise
            if verbose:
                print(f"  !! model emitted a malformed tool call, retrying: {e}")
            messages.append({"role": "user", "content": f"(system) Your last tool call failed: {e}. Try again with a valid call."})
            continue
        msg = resp.choices[0].message
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            return {"reply": msg.content, "escalated": False, "trace": trace}
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                        for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            result = dispatch(tc.function.name, tc.function.arguments)
            trace.append(tc.function.name)
            if verbose:
                print(f"  -> {tc.function.name}({tc.function.arguments}) = {json.dumps(result)}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return {"reply": "I couldn't fully resolve this automatically, so I've escalated your case to a human "
                     "support specialist who will contact you shortly.",
            "escalated": True, "trace": trace}


def chat():
    """Interactive session: the customer types messages; history is kept so they can follow up."""
    from groq import Groq
    client, history = Groq(), []
    print("Order Support Agent - describe your issue (type 'quit' to exit).")
    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {"quit", "exit", "bye"}:
            break
        if not text:
            continue
        out = run_agent(text, client=client, messages=history)
        print(f"\nAgent{' [ESCALATED]' if out['escalated'] else ''}: {out['reply']}")
        if out["escalated"]:
            break
    print("Goodbye!")


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set. Put your key in the .env file next to agent.py:\n"
                 "  GROQ_API_KEY=gsk_...")
    if len(sys.argv) > 1:  # one-shot: python agent.py "My order ORD-1002 arrived damaged"
        out = run_agent(" ".join(sys.argv[1:]))
        print(f"\nAgent{' [ESCALATED]' if out['escalated'] else ''}: {out['reply']}")
    else:                  # interactive: python agent.py
        chat()
