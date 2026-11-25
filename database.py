"""SQLite data layer: orders.db is created and seeded on first run; changes persist across runs.

Delete orders.db to reset to the sample data. Set ORDERS_DB to use a different file.
"""
import os
import sqlite3

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
    """Open the orders database, creating + seeding it on first use."""
    path = path or DB_PATH
    is_new = path == ":memory:" or not os.path.exists(path)
    db = sqlite3.connect(path, check_same_thread=False)  # used from tool worker threads; dispatcher serialises access
    db.row_factory = sqlite3.Row
    if is_new:
        db.executescript(SEED)
    return db


DB = init_db()  # tools read this via `database.DB`, so tests can swap in a fresh in-memory DB
