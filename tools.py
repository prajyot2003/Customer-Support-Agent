"""Step 1 + 2: the agent's tools (deterministic functions) and their declarative schemas.

Every tool returns a structured dict with "status": "success" | "error".
"""
import uuid

import database


def _order(order_id):
    return database.DB.execute("SELECT * FROM orders WHERE order_id = ?", (order_id.strip().upper(),)).fetchone()


def _stock(item_id):
    row = database.DB.execute("SELECT stock FROM inventory WHERE item_id = ?", (item_id,)).fetchone()
    return row["stock"] if row else 0


# --------------------------------------------------------------------------
# Step 1: Tools
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
    row = database.DB.execute("SELECT * FROM inventory WHERE item_id = ?", (item_id.strip().upper(),)).fetchone()
    if not row:
        return {"status": "error", "error": "ITEM_NOT_FOUND", "message": f"Unknown item id '{item_id}'."}
    return {"status": "success", "item_id": row["item_id"], "item_name": row["item_name"],
            "warehouse": row["warehouse"], "stock": row["stock"], "available": row["stock"] > 0}


def create_exchange(order_id):
    db = database.DB
    o = _order(order_id)
    if not o:
        return {"status": "error", "error": "ORDER_NOT_FOUND", "message": f"No order '{order_id}'."}
    # Policy enforced in code, not just in the prompt.
    if not o["exchange_eligible"]:
        return {"status": "error", "error": "NOT_ELIGIBLE", "message": "Order is not eligible for exchange; issue a refund instead."}
    if _stock(o["item_id"]) <= 0:
        return {"status": "error", "error": "OUT_OF_STOCK", "message": "No replacement units in stock; issue a refund instead."}
    if db.execute("SELECT 1 FROM exchanges WHERE order_id = ?", (o["order_id"],)).fetchone():
        return {"status": "error", "error": "ALREADY_EXCHANGED", "message": "An exchange already exists for this order."}
    exchange_id = "EXC-" + uuid.uuid4().hex[:8].upper()
    with db:
        db.execute("UPDATE inventory SET stock = stock - 1 WHERE item_id = ?", (o["item_id"],))
        db.execute("INSERT INTO exchanges VALUES (?, ?, ?)", (exchange_id, o["order_id"], o["item_id"]))
    return {"status": "success", "exchange_id": exchange_id, "order_id": o["order_id"], "item_id": o["item_id"]}


def process_refund(order_id, reason):
    db = database.DB
    o = _order(order_id)
    if not o:
        return {"status": "error", "error": "ORDER_NOT_FOUND", "message": f"No order '{order_id}'."}
    # Money path guard: refunds only when an exchange is impossible.
    if o["exchange_eligible"] and _stock(o["item_id"]) > 0:
        return {"status": "error", "error": "EXCHANGE_AVAILABLE",
                "message": "Replacement is in stock and order is eligible; create an exchange instead of refunding."}
    if db.execute("SELECT 1 FROM refunds WHERE order_id = ?", (o["order_id"],)).fetchone():
        return {"status": "error", "error": "ALREADY_REFUNDED", "message": "This order was already refunded."}
    refund_id = "RFD-" + uuid.uuid4().hex[:8].upper()
    with db:  # amount comes from the DB, never from the model
        db.execute("INSERT INTO refunds VALUES (?, ?, ?, ?)", (refund_id, o["order_id"], o["amount"], reason))
    return {"status": "success", "refund_id": refund_id, "order_id": o["order_id"],
            "amount_refunded": o["amount"], "method": "original payment method"}


def send_notification(email, message):
    """Simulated email: recorded in the notifications table instead of actually being sent."""
    if "@" not in email:
        return {"status": "error", "error": "INVALID_EMAIL", "message": f"'{email}' is not a valid email."}
    db = database.DB
    with db:
        cur = db.execute("INSERT INTO notifications (email, message) VALUES (?, ?)", (email, message))
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
