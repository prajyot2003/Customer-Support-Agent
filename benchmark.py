"""Measure the agent's cost: tokens, model calls, LLM latency, DB queries, cache hits.

    python3 benchmark.py before     # run all conversations, save benchmarks/before.json
    python3 benchmark.py after      # same, save benchmarks/after.json
    python3 benchmark.py compare    # print the before/after table (no API calls)

Each conversation runs on a fresh in-memory copy of the sample data, so orders.db is never touched.
"""
import json
import os
import sys
import time

import agent  # loads .env
import database
import tools

CONVERSATIONS = {
    "exchange": ["My headphones from order ORD-1001 arrived damaged."],
    "refund_out_of_stock": ["The kettle in order ORD-1002 is broken."],
    "refund_ineligible": ["Order ORD-1003, my lamp stopped working."],
    "invalid_order": ["My order ORD-0000 arrived damaged."],
    "two_orders": ["Both ORD-1001 and ORD-1002 arrived damaged."],
    "status_then_followup": ["Where is my order ORD-1001?", "Is it still eligible for an exchange?"],
    "multi_turn_refund": ["My kettle arrived broken.", "It's ORD-1002.", "When will I get the money?",
                          "Can you check ORD-1002 again and confirm the refund went through?"],
}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")
METRICS = ["model_calls", "prompt_tokens", "completion_tokens", "total_tokens", "llm_seconds",
           "db_queries", "cache_hits", "tool_calls"]


class MeteredClient:
    """Wraps the Groq client and records usage + latency of every chat completion."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        start = time.perf_counter()
        resp = self.inner.chat.completions.create(**kwargs)
        u = resp.usage
        self.calls.append({"prompt": u.prompt_tokens, "completion": u.completion_tokens,
                           "seconds": time.perf_counter() - start})
        return resp


def run_conversation(turns, groq_client):
    database.DB = database.init_db(":memory:")
    if hasattr(tools, "cache_clear"):
        tools.cache_clear()
    queries = []
    database.DB.set_trace_callback(lambda sql: queries.append(sql) if sql.lstrip().upper().startswith("SELECT") else None)
    client, history, trace = MeteredClient(groq_client), [], []
    for text in turns:
        trace += agent.run_agent(text, client=client, messages=history, verbose=False)["trace"]
    stats = getattr(tools, "CACHE_STATS", {"hits": 0})
    return {"model_calls": len(client.calls),
            "prompt_tokens": sum(c["prompt"] for c in client.calls),
            "completion_tokens": sum(c["completion"] for c in client.calls),
            "total_tokens": sum(c["prompt"] + c["completion"] for c in client.calls),
            "llm_seconds": round(sum(c["seconds"] for c in client.calls), 2),
            "db_queries": len(queries), "cache_hits": stats["hits"], "tool_calls": len(trace)}


def run(label):
    from groq import Groq
    groq_client = Groq()
    results = {}
    for name, turns in CONVERSATIONS.items():
        results[name] = run_conversation(turns, groq_client)
        print(f"{name:22} {results[name]}", flush=True)
    totals = {m: round(sum(r[m] for r in results.values()), 2) for m in METRICS}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{label}.json"), "w") as f:
        json.dump({"model": agent.MODEL, "conversations": results, "totals": totals}, f, indent=2)
    print(f"TOTAL {totals}")


def compare():
    before, after = (json.load(open(os.path.join(OUT_DIR, f"{n}.json"))) for n in ("before", "after"))
    print(f"| Metric (all {len(CONVERSATIONS)} conversations) | Before | After | Change |\n|---|---|---|---|")
    for m in METRICS:
        b, a = before["totals"][m], after["totals"][m]
        change = f"{(a - b) / b * 100:+.0f}%" if b else "n/a"
        print(f"| {m} | {b} | {a} | {change} |")
    print("\n| Conversation | Tokens before | Tokens after | Calls before | Calls after |\n|---|---|---|---|---|")
    for name in CONVERSATIONS:
        b, a = before["conversations"][name], after["conversations"][name]
        print(f"| {name} | {b['total_tokens']} | {a['total_tokens']} | {b['model_calls']} | {a['model_calls']} |")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    compare() if mode == "compare" else run(mode)
