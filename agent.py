"""Decision-Making Resolution Agent: Step 4 + 5, the ReAct loop, guardrails and chat CLI.

Run:   python agent.py                                      (interactive chat)
       python agent.py "My order ORD-1002 arrived damaged"  (one-shot)
Needs: pip install groq ; GROQ_API_KEY in .env (or exported in the shell)

Layout: database.py (SQLite) -> tools.py (tools + schemas) -> dispatcher.py (whitelist) -> agent.py (loop)
"""
import json
import os
import sys

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
MAX_ITERATIONS = 6  # lookup, inventory, exchange/refund, notify, final answer + 1 retry

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
    if len(sys.argv) > 1:
        out = run_agent(" ".join(sys.argv[1:]))
        print(f"\nAgent{' [ESCALATED]' if out['escalated'] else ''}: {out['reply']}")
    else:
        chat()
