# Customer Support Resolution Agent

An autonomous, tool-using AI agent that resolves damaged-item complaints. It reads order records, checks warehouse stock, and decides on its own whether to **exchange** the item or **refund** the customer, then sends a confirmation email.

Built with the [Groq](https://groq.com) API (OpenAI-compatible tool calling) and a hand-written ReAct loop. There is no agent framework, so every step is visible in the code, one file per stage (see [Project structure](#project-structure)).

## How it works

```
Customer message
      │
      ▼
┌─────────────┐   tool call    ┌──────────────────┐
│  LLM (Groq) │ ─────────────► │ Safe dispatcher  │──► lookup_order
│             │ ◄───────────── │ (whitelist +     │──► check_inventory
└─────────────┘  tool result   │  arg validation) │──► create_exchange
      │                        └──────────────────┘──► process_refund
      │ final answer                                 ──► send_notification
      ▼
 Reply to customer          (loop ends after 10 rounds → escalate to a human)
```

**Decision logic**

| Situation | Outcome |
|---|---|
| Eligible for exchange **and** stock > 0 | Exchange (stock is reserved) |
| Stock = 0 | Refund to original payment method |
| Not eligible (e.g. outside 30-day window) | Refund to original payment method |
| Order ID not found | Asks the customer to verify the ID |

The model is never told which tools to call for a given request. It works that out from the tool schemas and the results it reads back.

## Features

- **5 tools** with structured `{"status": "success" | "error", ...}` results
- **Declarative JSON schemas** with descriptions, typed parameters, and required fields
- **Whitelisted dispatcher.** Unknown tool names are rejected with `UNAUTHORIZED_TOOL`.
- **Argument validation.** Malformed JSON, missing, unknown, wrong-type, or empty arguments are returned to the model so it can correct itself.
- **Iteration ceiling.** The agent escalates to a human if a message isn't resolved within 10 rounds, and tells the customer about any refund or exchange already made.
- **Policy enforced in code, not just the prompt**:
  - refunds are refused when an exchange is possible
  - exchanges are refused when out of stock or ineligible
  - one resolution per order: no double refunds, and no refund after an exchange (or vice versa)
  - the refund amount always comes from the database, never from the model
- **Persistent SQLite database** (`orders.db`), created and seeded on first run
- **Interactive chat** with conversation memory, so the customer can correct a wrong order ID

## Setup

Requires Python 3.9+.

```bash
git clone https://github.com/prajyot2003/Customer-Support-Agent.git
cd Customer-Support-Agent
pip install groq
```

Create a `.env` file in the project folder with your [Groq API key](https://console.groq.com/keys):

```
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

`.env` is git-ignored, so never commit it. Variables exported in your shell take priority over `.env`.

## Usage

**Interactive chat**

```bash
python3 agent.py
```

```
Order Support Agent - describe your issue (type 'quit' to exit).

You: My order ORD-1002 arrived damaged
  -> lookup_order({"order_id":"ORD-1002"}) = {...}
  -> check_inventory({"item_id":"SKU-KETTLE"}) = {"stock": 0, ...}
  -> process_refund({"order_id":"ORD-1002", ...}) = {"refund_id": "RFD-...", "amount_refunded": 34.5, ...}
  -> send_notification({"email":"arjun@example.com", ...}) = {...}

Agent: Your refund of $34.50 has been processed to your original payment method...
```

**One-shot**

```bash
python3 agent.py "My headphones from order ORD-1001 arrived broken"
```

## Sample data

On first run, `orders.db` is created with this sample data (fictional customers):

| Order | Item | Stock | Exchange eligible | Expected result |
|---|---|---|---|---|
| ORD-1001 | Wireless Headphones | 12 | Yes | Exchange |
| ORD-1002 | Electric Kettle | 0 | Yes | Refund (out of stock) |
| ORD-1003 | Desk Lamp | 5 | No (outside 30 days) | Refund (ineligible) |
| ORD-0000 | n/a | n/a | n/a | "Order not found, please verify" |

Changes persist between runs. To reset the data:

```bash
rm orders.db
```

To inspect it:

```bash
sqlite3 orders.db "select * from orders; select * from refunds; select * from exchanges;"
```

## Tests

```bash
python3 test_agent.py
```

- **14 offline tests** need no API key. They cover the dispatcher whitelist, argument validation, the refund and exchange policy guards, one-resolution-per-order, order ID normalization, reply cleanup, the ReAct loop driven by a scripted fake model, max-iteration escalation, and multi-turn memory.
- **4 live tests** run against Groq when `GROQ_API_KEY` is set. They check the 4 acceptance scenarios in the sample-data table, including the exact order of tool calls.

Tests use a fresh in-memory database and never modify `orders.db`.

## Project structure

| File | Assignment step | What it contains |
|---|---|---|
| [`database.py`](database.py) | Data | SQLite schema, sample data, `init_db()` |
| [`tools.py`](tools.py) | Steps 1 + 2 | The 5 tool functions and their JSON schemas (`TOOLS`) |
| [`dispatcher.py`](dispatcher.py) | Step 3 | `REGISTRY` whitelist and `dispatch()` with argument validation |
| [`agent.py`](agent.py) | Steps 4 + 5 | ReAct loop, iteration ceiling and escalation, `.env` loading, chat CLI |
| [`test_agent.py`](test_agent.py) | Acceptance | Offline + live tests |
| `.env` | | Your API key (create it; git-ignored) |
| `orders.db` | | SQLite data (auto-created; git-ignored) |

Dependencies flow one way: `database.py` → `tools.py` → `dispatcher.py` → `agent.py`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | (none, required) | Groq API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Any Groq model that supports tool calling |
| `ORDERS_DB` | `./orders.db` | Path to the SQLite database |
