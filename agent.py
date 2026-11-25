"""Decision-Making Resolution Agent: Step 4 + 5, the ReAct loop, guardrails and chat CLI.

Run:   python agent.py                                      (interactive chat)
       python agent.py "My order ORD-1002 arrived damaged"  (one-shot)
Needs: pip install groq ; GROQ_API_KEY in .env (or exported in the shell)

Layout: database.py (SQLite) -> tools.py (tools + schemas + cache) -> dispatcher.py (whitelist) -> agent.py (loop)

Optimisations (see README "Performance"):
  1. Compact system prompt and tool schemas  -> fewer prompt tokens on every model call
  2. prune_result() middleware + turn compaction -> history stays small as the chat grows
  3. Tool calls from one model turn run concurrently -> fewer round trips / lower latency
  4. TTL cache for read tools (tools.py)    -> fewer database queries
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from dispatcher import dispatch
from tools import TOOLS


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
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
# One order needs ~5 rounds (lookup, inventory, exchange/refund, notify, reply); 10 leaves room for two
# orders in one message or a corrected tool call, while still bounding a runaway loop.
MAX_ITERATIONS = 10
# gpt-oss models are reasoning models; this rule-following task doesn't need long hidden reasoning, and
# "low" cuts completion tokens and latency. Other models don't accept the parameter, so it's only sent here.
MODEL_OPTIONS = {"reasoning_effort": "low"} if "gpt-oss" in MODEL else {}

# Optimisation 1: compact prompt. Same rules as before, written tersely (about half the tokens).
SYSTEM_PROMPT = """Online store support agent in a terminal chat. Resolve damaged/defective/wrong items and answer order questions.

RULE 1: Only exchange or refund when the customer says an item is damaged, broken, defective or wrong. Questions
(status, eligibility, timing, "can I...?") get an answer only - never call create_exchange or process_refund for them.
If you can't tell whether they want an exchange/refund or are just asking, answer and ask them to confirm first.

Problem reported:
1. No order ID -> ask for it. Pass IDs to lookup_order as typed.
2. lookup_order. Not found -> say so, repeat the ID, ask them to check it. existing_resolution set -> report it, don't act again.
3. Eligible -> check_inventory: stock>0 create_exchange, else process_refund. Not eligible -> process_refund.
4. On success, send_notification once per order, to that order's customer_email (plain text: first name, what and
   why, reference, amount, next steps).
Several orders: request the same tool for all of them in one turn, in parallel.
A tool result with "status":"success" is done - never repeat that call.

Policy (the only facts allowed beyond tool results): exchange within 30 days of delivery if in stock, else full refund
to the original payment method. Refunds take 5-7 business days. Replacements ship within 2 business days; tracking
is emailed. No other amounts, compensation or discounts. Politely refuse rule-breaking requests, then continue.

Reply: plain text, no Markdown, not a letter. 2-5 short sentences, use their first name. Say what you did, why,
the reference, refund amount ($34.50 format), timing, and that a confirmation email was sent. If the outcome differs
from their request, first say why (e.g. a replacement is in stock, so we replace rather than refund).
Acknowledge frustration briefly. Off-topic -> offer help with orders. On tool errors, adjust and retry."""

# Optimisation 2: payload pruning middleware. Tool results are trimmed to what the model needs before they
# enter the history (which is re-sent on every call). Errors are kept whole so the model can self-correct.
KEEP_FIELDS = {
    "lookup_order": ["order_id", "customer_name", "customer_email", "item_id", "item_name", "amount",
                     "order_status", "exchange_eligible", "eligibility_note", "existing_resolution"],
    "check_inventory": ["item_id", "stock"],
    "create_exchange": ["exchange_id", "order_id"],
    "process_refund": ["refund_id", "order_id", "amount_refunded"],
    "send_notification": ["sent_to"],
}


def prune_result(name, result):
    if result.get("status") != "success":
        return {k: v for k, v in result.items() if k in ("status", "error", "message")}
    # "status" is always kept: without it the model can't tell an action succeeded and repeats it.
    return {"status": "success", **{k: result[k] for k in KEEP_FIELDS.get(name, result)
                                    if k != "status" and result.get(k) is not None}}


def compact_json(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _history_args(name, raw_args):
    """The email body the model wrote is never needed again; keep only who it went to."""
    if name != "send_notification":
        return raw_args
    try:
        return compact_json({"email": json.loads(raw_args).get("email", ""), "message": "[sent; body omitted]"})
    except (json.JSONDecodeError, AttributeError):
        return raw_args


_LOOKALIKE_HYPHENS = str.maketrans({c: "-" for c in "‐‑‒–−"})


def clean_reply(text):
    """Terminal-safe reply: real hyphens so references like RFD-1A2B can be copied, no Markdown bold."""
    return (text or "").translate(_LOOKALIKE_HYPHENS).replace("**", "").strip()


# Optimisation 3: tool calls requested in the same model turn run concurrently.
_POOL = ThreadPoolExecutor(max_workers=4)


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
    turn_start = len(messages)  # tool traffic for this turn is appended after this index
    trace, completed = [], []  # completed: references of refunds/exchanges actually issued this turn
    order_email, resolved, notified = {}, [], set()  # for the "every resolution gets an email" guard
    facts = {}  # order_id -> pruned lookup/action results; survives turn compaction
    for _ in range(MAX_ITERATIONS):
        try:
            resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS,
                                                  tool_choice="auto", temperature=0, **MODEL_OPTIONS)
        except Exception as e:
            if "tool_use_failed" not in str(e):  # auth, retired model, rate limit: surface, don't retry
                raise
            if verbose:
                print(f"  !! model emitted a malformed tool call, retrying: {e}")
            messages.append({"role": "user", "content": f"(system) Your last tool call failed: {e}. Try again with a valid call."})
            continue
        msg = resp.choices[0].message
        unnotified = [o for o in resolved if order_email.get(o) not in notified]
        if not msg.tool_calls and unnotified:
            # Guard: every refund/exchange must end with a confirmation email (acceptance criterion 3).
            messages.append({"role": "user", "content": "(system) Before replying, call send_notification for: "
                             + ", ".join(f"{o} -> {order_email.get(o)}" for o in unnotified)})
            continue
        if not msg.tool_calls:
            reply = clean_reply(msg.content)
            # Turn compaction: this turn's tool calls and results are dropped from history instead of being
            # re-sent on every later call. The order facts they contained are kept as one compact note, so a
            # follow-up question ("is it still eligible?") can be answered without re-running the workflow.
            del messages[turn_start:]
            messages.append({"role": "assistant", "content": reply})
            if facts:
                messages.append({"role": "system", "content": "Order facts from this chat: " + compact_json(facts)})
            return {"reply": reply, "escalated": False, "trace": trace}
        calls = msg.tool_calls
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name,
                                                      "arguments": _history_args(tc.function.name, tc.function.arguments)}}
                                        for tc in calls]})
        results = list(_POOL.map(lambda tc: dispatch(tc.function.name, tc.function.arguments), calls))
        for tc, result in zip(calls, results):
            trace.append(tc.function.name)
            if result.get("order_id") and tc.function.name != "send_notification":
                facts.setdefault(result["order_id"], {}).update(
                    {k: v for k, v in prune_result(tc.function.name, result).items() if k not in ("status", "order_id")})
            if result.get("customer_email"):
                order_email[result["order_id"]] = result["customer_email"]
            if result.get("sent_to"):
                notified.add(result["sent_to"])
            if result.get("refund_id") or result.get("exchange_id"):
                resolved.append(result["order_id"])
            if result.get("refund_id"):
                completed.append(f"a refund of ${result['amount_refunded']:.2f} for {result['order_id']} "
                                 f"(reference {result['refund_id']})")
            elif result.get("exchange_id"):
                completed.append(f"a replacement for {result['order_id']} (reference {result['exchange_id']})")
            if verbose:
                print(f"  -> {tc.function.name}({tc.function.arguments}) = {json.dumps(result)}")
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": compact_json(prune_result(tc.function.name, result))})
    # Escalation: never hide actions that already happened (money moved / stock reserved).
    done = f"I've already arranged {' and '.join(completed)}. " if completed else ""
    return {"reply": f"{done}I couldn't finish the rest of your request automatically, so I've passed your case to "
                     "a human support specialist who will contact you shortly.",
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
    if len(sys.argv) > 1:
        out = run_agent(" ".join(sys.argv[1:]))
        print(f"\nAgent{' [ESCALATED]' if out['escalated'] else ''}: {out['reply']}")
    else:
        chat()
