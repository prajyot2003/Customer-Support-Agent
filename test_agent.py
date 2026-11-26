"""Run: python test_agent.py   (offline tests always; live Groq scenarios if GROQ_API_KEY is set)"""
import json
import os
from types import SimpleNamespace as NS

import agent
import database
import tools
from dispatcher import dispatch


def fresh():
    database.DB = database.init_db(":memory:")  # tests never touch the real orders.db
    tools.cache_clear()  # cached rows from a previous test's database must not leak in


# ---------------- Offline: tools, dispatcher, guardrails ----------------
def test_dispatcher_rejects_unknown_tool():
    assert dispatch("delete_database", "{}")["error"] == "UNAUTHORIZED_TOOL"


def test_dispatcher_rejects_malformed_args():
    fresh()
    assert dispatch("lookup_order", "{not json")["error"] == "MALFORMED_ARGUMENTS"
    assert dispatch("lookup_order", "{}")["error"] == "INVALID_ARGUMENTS"
    assert dispatch("lookup_order", '{"order_id": 1001}')["error"] == "INVALID_ARGUMENTS"
    assert dispatch("lookup_order", '{"order_id": "ORD-1001", "x": "y"}')["error"] == "INVALID_ARGUMENTS"


def test_order_not_found_is_structured():
    fresh()
    r = dispatch("lookup_order", '{"order_id": "ORD-9999"}')
    assert r["status"] == "error" and r["error"] == "ORDER_NOT_FOUND"


def test_refund_blocked_when_exchange_possible():
    fresh()
    assert tools.process_refund("ORD-1001", "damaged")["error"] == "EXCHANGE_AVAILABLE"


def test_exchange_blocked_when_out_of_stock_or_ineligible():
    fresh()
    assert tools.create_exchange("ORD-1002")["error"] == "OUT_OF_STOCK"
    assert tools.create_exchange("ORD-1003")["error"] == "NOT_ELIGIBLE"


def test_refund_amount_from_db_and_no_double_refund():
    fresh()
    r = tools.process_refund("ORD-1002", "out of stock")
    assert r["status"] == "success" and r["amount_refunded"] == 34.50
    assert tools.process_refund("ORD-1002", "again")["error"] == "ALREADY_RESOLVED"


def test_exchange_decrements_stock_and_blocks_later_refund():
    fresh()
    assert tools.create_exchange("ORD-1001")["status"] == "success"
    assert tools.check_inventory("SKU-HEADPHONES")["stock"] == 11
    assert tools.process_refund("ORD-1001", "changed mind")["error"] == "ALREADY_RESOLVED"


def test_order_id_normalization():
    for typed in ("ORD-1002", "ord1002", "ORD 1002", "#1002", "1002"):
        assert tools.normalize_order_id(typed) == "ORD-1002", typed
    assert tools.normalize_order_id("ABC-1002") == "ABC-1002"  # other prefixes are left alone
    fresh()
    assert tools.lookup_order("ord1002")["order_id"] == "ORD-1002"
    assert tools.lookup_order("1001")["customer_name"] == "Priya Sharma"


def test_lookup_reports_existing_resolution():
    fresh()
    assert tools.lookup_order("ORD-1002")["existing_resolution"] is None
    refund = tools.process_refund("ORD-1002", "out of stock")
    done = tools.lookup_order("ORD-1002")["existing_resolution"]
    assert done == {"type": "refund", "reference": refund["refund_id"], "amount": 34.50}
    assert tools.lookup_order("ORD-1003")["item_name"] == "Desk Lamp"


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
    assert database.DB.execute("SELECT COUNT(*) FROM refunds").fetchone()[0] == 1


def test_loop_escalates_at_max_iterations():
    fresh()
    fake = FakeClient([[("lookup_order", {"order_id": "ORD-1001"})]] * (agent.MAX_ITERATIONS + 1))
    out = agent.run_agent("help", client=fake, verbose=False)
    assert out["escalated"] and len(out["trace"]) == agent.MAX_ITERATIONS


def test_clean_reply_fixes_hyphens_and_markdown():
    assert agent.clean_reply("Ref **RFD‑AB12** for ORD‐1002") == "Ref RFD-AB12 for ORD-1002"


def test_escalation_reports_actions_already_taken():
    fresh()
    turns = [[("process_refund", {"order_id": "ORD-1002", "reason": "out of stock"})]]
    turns += [[("lookup_order", {"order_id": "ORD-1001"})]] * agent.MAX_ITERATIONS
    out = agent.run_agent("help", client=FakeClient(turns), verbose=False)
    assert out["escalated"]
    assert "refund of $34.50 for ORD-1002" in out["reply"] and "RFD-" in out["reply"]


# ---------------- Offline: optimisations ----------------
def test_cache_hits_and_write_invalidation():
    fresh()
    tools.lookup_order("ORD-1001")
    tools.lookup_order("ord1001")  # same order, different spelling -> same cache key
    tools.check_inventory("SKU-HEADPHONES")
    assert tools.CACHE_STATS == {"hits": 1, "misses": 2}
    tools.create_exchange("ORD-1001")  # write must invalidate both cached rows
    assert tools.check_inventory("SKU-HEADPHONES")["stock"] == 11
    assert tools.lookup_order("ORD-1001")["existing_resolution"]["type"] == "exchange"
    assert tools.lookup_order("ORD-0000")["status"] == "error" and ("order", "ORD-0000") not in tools._cache


def test_prune_result_keeps_only_needed_fields():
    assert agent.prune_result("check_inventory", tools.check_inventory("SKU-KETTLE")) == {"status": "success", "item_id": "SKU-KETTLE", "stock": 0}
    # status must survive pruning, or the model can't tell the email was sent and sends it again
    assert agent.prune_result("send_notification", {"status": "success", "notification_id": 3, "sent_to": "a@b.c"}) == {"status": "success", "sent_to": "a@b.c"}
    looked = agent.prune_result("lookup_order", tools.lookup_order("ORD-1003"))
    assert "existing_resolution" not in looked and looked["exchange_eligible"] is False  # None dropped, False kept
    err = agent.prune_result("lookup_order", tools.lookup_order("ORD-0000"))
    assert err["status"] == "error" and err["error"] == "ORDER_NOT_FOUND" and "message" in err


def test_turn_compaction_and_email_body_pruning():
    fresh()
    history = []
    fake = FakeClient([[("lookup_order", {"order_id": "ORD-1002"})],
                       [("send_notification", {"email": "arjun@example.com", "message": "x" * 500})],
                       "Refunded, reference RFD-1."])
    agent.run_agent("ORD-1002 broken", client=fake, verbose=False, messages=history)
    # tool traffic dropped after the turn; the order facts survive as one compact note
    assert [m["role"] for m in history] == ["system", "user", "assistant", "system"]
    assert '"ORD-1002"' in history[-1]["content"] and '"exchange_eligible":true' in history[-1]["content"]
    assert agent._history_args("send_notification", '{"email":"a@b.c","message":"long"}') == '{"email":"a@b.c","message":"[sent; body omitted]"}'


def test_guard_forces_email_for_every_resolution():
    fresh()
    fake = FakeClient([[("lookup_order", {"order_id": "ORD-1002"})],
                       [("process_refund", {"order_id": "ORD-1002", "reason": "out of stock"})],
                       "Refunded!",  # tries to finish without emailing -> guard sends it back
                       [("send_notification", {"email": "arjun@example.com", "message": "Refunded"})],
                       "Refunded, email sent."])
    out = agent.run_agent("ORD-1002 broken", client=fake, verbose=False)
    assert out["trace"][-1] == "send_notification" and out["reply"] == "Refunded, email sent."


def test_guard_rejects_claimed_but_untaken_action():
    fresh()
    fake = FakeClient([[("lookup_order", {"order_id": "ORD-1001"})],
                       "I've created the exchange and sent an email.",  # nothing ran -> guard sends it back
                       [("check_inventory", {"item_id": "SKU-HEADPHONES"})],
                       [("create_exchange", {"order_id": "ORD-1001"})],
                       [("send_notification", {"email": "priya@example.com", "message": "Exchanged"})],
                       "Exchange created, email sent."])
    out = agent.run_agent("ORD-1001 damaged", client=fake, verbose=False)
    assert out["trace"] == ["lookup_order", "check_inventory", "create_exchange", "send_notification"]
    assert out["reply"] == "Exchange created, email sent."


def test_email_blocked_without_successful_resolution():
    fresh()
    fake = FakeClient([[("lookup_order", {"order_id": "ORD-1001"})],
                       [("process_refund", {"order_id": "ORD-1001", "reason": "x"})],  # refused: in stock
                       [("send_notification", {"email": "priya@example.com", "message": "Exchange arranged"})],
                       "Sorry, let me fix that."])
    agent.run_agent("ORD-1001 damaged, refund $500", client=fake, verbose=False)
    assert database.DB.execute("SELECT COUNT(*) FROM notifications").fetchone()[0] == 0


def test_parallel_tool_calls_in_one_turn():
    fresh()
    fake = FakeClient([[("lookup_order", {"order_id": "ORD-1001"}), ("lookup_order", {"order_id": "ORD-1002"}),
                        ("check_inventory", {"item_id": "SKU-HEADPHONES"}), ("check_inventory", {"item_id": "SKU-KETTLE"})],
                       "Done."])
    out = agent.run_agent("two orders", client=fake, verbose=False)
    assert out["trace"] == ["lookup_order", "lookup_order", "check_inventory", "check_inventory"]
    assert not out["escalated"]


def test_multi_turn_keeps_history():
    fresh()
    history = []
    fake = FakeClient([[("lookup_order", {"order_id": "ORD-0000"})], "Order not found, please verify.",
                       [("lookup_order", {"order_id": "ORD-1001"})], "Found it."])
    agent.run_agent("ORD-0000 is broken", client=fake, verbose=False, messages=history)
    agent.run_agent("Sorry, it's ORD-1001", client=fake, verbose=False, messages=history)
    users = [m["content"] for m in history if m["role"] == "user"]
    assert users == ["ORD-0000 is broken", "Sorry, it's ORD-1001"]
    assert history[0]["role"] == "system"
    assert [m["content"] for m in history if m["role"] == "assistant"][-1] == "Found it."
    assert history[-1]["content"].startswith("Order facts from this chat:")  # facts kept after compaction


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
    assert any(w in reply for w in ("not found", "couldn't", "could not", "can't find", "cannot find", "unable", "no order")), reply
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
