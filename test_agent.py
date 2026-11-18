"""Run: python test_agent.py   (offline tests always; live Groq scenarios if GROQ_API_KEY is set)"""
import json
import os
from types import SimpleNamespace as NS

import agent


def fresh():
    agent.DB = agent.init_db(":memory:")  # tests never touch the real orders.db


# ---------------- Offline: tools, dispatcher, guardrails ----------------
def test_dispatcher_rejects_unknown_tool():
    assert agent.dispatch("delete_database", "{}")["error"] == "UNAUTHORIZED_TOOL"


def test_dispatcher_rejects_malformed_args():
    fresh()
    assert agent.dispatch("lookup_order", "{not json")["error"] == "MALFORMED_ARGUMENTS"
    assert agent.dispatch("lookup_order", "{}")["error"] == "INVALID_ARGUMENTS"
    assert agent.dispatch("lookup_order", '{"order_id": 1001}')["error"] == "INVALID_ARGUMENTS"
    assert agent.dispatch("lookup_order", '{"order_id": "ORD-1001", "x": "y"}')["error"] == "INVALID_ARGUMENTS"


def test_order_not_found_is_structured():
    fresh()
    r = agent.dispatch("lookup_order", '{"order_id": "ORD-9999"}')
    assert r["status"] == "error" and r["error"] == "ORDER_NOT_FOUND"


def test_refund_blocked_when_exchange_possible():
    fresh()
    assert agent.process_refund("ORD-1001", "damaged")["error"] == "EXCHANGE_AVAILABLE"


def test_exchange_blocked_when_out_of_stock_or_ineligible():
    fresh()
    assert agent.create_exchange("ORD-1002")["error"] == "OUT_OF_STOCK"
    assert agent.create_exchange("ORD-1003")["error"] == "NOT_ELIGIBLE"


def test_refund_amount_from_db_and_no_double_refund():
    fresh()
    r = agent.process_refund("ORD-1002", "out of stock")
    assert r["status"] == "success" and r["amount_refunded"] == 34.50
    assert agent.process_refund("ORD-1002", "again")["error"] == "ALREADY_REFUNDED"


def test_exchange_decrements_stock():
    fresh()
    assert agent.create_exchange("ORD-1001")["status"] == "success"
    assert agent.check_inventory("SKU-HEADPHONES")["stock"] == 11


# ---------------- Offline: ReAct loop with a scripted fake model ----------------
class FakeClient:
    """Replays a scripted list of model turns: each is a list of (tool, args) or a final string."""
    def __init__(self, turns):
        self.turns = iter(turns)
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **_):
        turn = next(self.turns)
        if isinstance(turn, str):
            msg = NS(content=turn, tool_calls=None)
        else:
            msg = NS(content="", tool_calls=[NS(id=f"c{i}", function=NS(name=n, arguments=json.dumps(a)))
                                             for i, (n, a) in enumerate(turn)])
        return NS(choices=[NS(message=msg)])


def test_loop_executes_tools_and_feeds_results_back():
    fresh()
    fake = FakeClient([[("lookup_order", {"order_id": "ORD-1002"})],
                       [("check_inventory", {"item_id": "SKU-KETTLE"})],
                       [("process_refund", {"order_id": "ORD-1002", "reason": "out of stock"})],
                       [("send_notification", {"email": "arjun@example.com", "message": "Refunded"})],
                       "Your refund has been issued."])
    out = agent.run_agent("ORD-1002 damaged", client=fake, verbose=False)
    assert out["trace"] == ["lookup_order", "check_inventory", "process_refund", "send_notification"]
    assert not out["escalated"]
    assert agent.DB.execute("SELECT COUNT(*) FROM refunds").fetchone()[0] == 1


def test_loop_escalates_at_max_iterations():
    fresh()
    fake = FakeClient([[("lookup_order", {"order_id": "ORD-1001"})]] * (agent.MAX_ITERATIONS + 1))
    out = agent.run_agent("help", client=fake, verbose=False)
    assert out["escalated"] and len(out["trace"]) == agent.MAX_ITERATIONS


def test_multi_turn_keeps_history():
    fresh()
    history = []
    fake = FakeClient([[("lookup_order", {"order_id": "ORD-0000"})], "Order not found, please verify.",
                       [("lookup_order", {"order_id": "ORD-1001"})], "Found it."])
    agent.run_agent("ORD-0000 is broken", client=fake, verbose=False, messages=history)
    agent.run_agent("Sorry, it's ORD-1001", client=fake, verbose=False, messages=history)
    users = [m["content"] for m in history if m["role"] == "user"]
    assert users == ["ORD-0000 is broken", "Sorry, it's ORD-1001"]
    assert history[0]["role"] == "system" and history[-1] == {"role": "assistant", "content": "Found it."}


# ---------------- Live: acceptance criteria against Groq ----------------
def _live(order_msg):
    fresh()
    out = agent.run_agent(order_msg, verbose=True)
    print(f"  REPLY: {out['reply']}\n")
    return out


def live_exchange_when_in_stock():
    out = _live("My headphones from order ORD-1001 arrived damaged. Please help.")
    assert out["trace"][:2] == ["lookup_order", "check_inventory"], out["trace"]
    assert "create_exchange" in out["trace"] and "process_refund" not in out["trace"]
    assert "send_notification" in out["trace"] and not out["escalated"]


def live_refund_when_out_of_stock():
    out = _live("The kettle in order ORD-1002 is broken, I want a replacement.")
    assert out["trace"][:2] == ["lookup_order", "check_inventory"], out["trace"]
    assert "process_refund" in out["trace"] and "create_exchange" not in out["trace"]
    assert "send_notification" in out["trace"] and not out["escalated"]


def live_refund_when_ineligible():
    out = _live("Order ORD-1003, my lamp stopped working. Can I exchange it?")
    assert "process_refund" in out["trace"] and "create_exchange" not in out["trace"]
    assert "send_notification" in out["trace"]


def live_invalid_order_id():
    out = _live("My order ORD-0000 arrived damaged.")
    assert out["trace"] == ["lookup_order"] and not out["escalated"]
    reply = out["reply"].lower().replace("’", "'").replace("‑", "-")  # LLMs emit curly quotes / nb-hyphens
    assert any(w in reply for w in ("not found", "couldn't", "could not", "unable", "no order")), reply
    assert any(w in reply for w in ("verify", "check", "confirm", "correct")), reply


if __name__ == "__main__":
    g = dict(globals())
    for name in [n for n in g if n.startswith("test_")]:
        g[name]()
        print(f"PASS {name}")
    if os.environ.get("GROQ_API_KEY"):
        for name in [n for n in g if n.startswith("live_")]:
            print(f"--- {name}")
            g[name]()
            print(f"PASS {name}")
    else:
        print("SKIP live Groq scenarios (set GROQ_API_KEY to run them)")
