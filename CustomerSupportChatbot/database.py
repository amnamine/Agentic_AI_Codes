import streamlit as st
import random

# Default mock data
DEFAULT_ACCOUNTS = {
    "alice@example.com": {
        "name": "Alice Smith",
        "email": "alice@example.com",
        "status": "Active",
        "created_at": "2025-01-15",
        "newsletter": True
    },
    "bob@example.com": {
        "name": "Bob Johnson",
        "email": "bob@example.com",
        "status": "Suspended",
        "created_at": "2024-11-20",
        "newsletter": False
    }
}

DEFAULT_ORDERS = {
    "ORD-88231": {
        "order_number": "ORD-88231",
        "email": "alice@example.com",
        "items": "1x Wireless Headphones, 1x USB-C Cable",
        "total": 129.98,
        "status": "Delivered",
        "shipping_address": "123 Main St, Springfield, IL 62701",
        "tracking_number": "TRK-99211029",
        "cancellation_fee": 0.0,
        "refund_status": "None"
    },
    "ORD-11502": {
        "order_number": "ORD-11502",
        "email": "alice@example.com",
        "items": "1x Mechanical Keyboard",
        "total": 89.99,
        "status": "Shipped",
        "shipping_address": "123 Main St, Springfield, IL 62701",
        "tracking_number": "TRK-88120481",
        "cancellation_fee": 5.0,
        "refund_status": "None"
    },
    "ORD-45691": {
        "order_number": "ORD-45691",
        "email": "bob@example.com",
        "items": "2x Gaming Mouse Pad",
        "total": 39.98,
        "status": "Pending",
        "shipping_address": "456 Oak Ave, Metropolis, NY 10001",
        "tracking_number": "TRK-77123951",
        "cancellation_fee": 0.0,
        "refund_status": "None"
    }
}

def init_db():
    if "accounts" not in st.session_state:
        st.session_state.accounts = DEFAULT_ACCOUNTS.copy()
    if "orders" not in st.session_state:
        st.session_state.orders = DEFAULT_ORDERS.copy()

def get_order(order_number):
    init_db()
    # Normalize input
    order_number = str(order_number).strip().upper()
    if not order_number.startswith("ORD-") and order_number.isdigit():
        order_number = f"ORD-{order_number}"
    return st.session_state.orders.get(order_number)

def cancel_order(order_number):
    init_db()
    order = get_order(order_number)
    if not order:
        return False, "Order not found."
    if order["status"] == "Cancelled":
        return False, "Order is already cancelled."
    if order["status"] == "Delivered":
        return False, "Order has already been delivered and cannot be cancelled."
    
    order["status"] = "Cancelled"
    st.session_state.orders[order["order_number"]] = order
    return True, f"Order {order['order_number']} has been successfully cancelled."

def update_shipping_address(order_number, new_address):
    init_db()
    order = get_order(order_number)
    if not order:
        return False, "Order not found."
    if order["status"] in ["Delivered", "Cancelled"]:
        return False, f"Cannot update address. Order is already {order['status'].lower()}."
    
    order["shipping_address"] = new_address
    st.session_state.orders[order["order_number"]] = order
    return True, f"Shipping address for order {order['order_number']} has been updated to: {new_address}."

def get_account(email):
    init_db()
    return st.session_state.accounts.get(email.strip().lower())

def create_account(name, email):
    init_db()
    email_clean = email.strip().lower()
    if email_clean in st.session_state.accounts:
        return False, "Account with this email already exists."
    
    st.session_state.accounts[email_clean] = {
        "name": name,
        "email": email_clean,
        "status": "Active",
        "created_at": "2026-07-20", # Current simulated year
        "newsletter": True
    }
    return True, f"Account successfully created for {name} ({email_clean})."

def delete_account(email):
    init_db()
    email_clean = email.strip().lower()
    if email_clean not in st.session_state.accounts:
        return False, "Account not found."
    
    del st.session_state.accounts[email_clean]
    return True, f"Account for {email_clean} has been successfully deleted."

def update_subscription(email, subscribe=True):
    init_db()
    email_clean = email.strip().lower()
    account = get_account(email_clean)
    if not account:
        # Create a basic account placeholder to allow subscription management
        st.session_state.accounts[email_clean] = {
            "name": "Customer",
            "email": email_clean,
            "status": "Active",
            "created_at": "2026-07-20",
            "newsletter": subscribe
        }
        status_str = "subscribed to" if subscribe else "unsubscribed from"
        return True, f"Email {email_clean} is now {status_str} the newsletter."
    
    account["newsletter"] = subscribe
    st.session_state.accounts[email_clean] = account
    status_str = "subscribed to" if subscribe else "unsubscribed from"
    return True, f"Email {email_clean} is now {status_str} the newsletter."

def request_refund(order_number):
    init_db()
    order = get_order(order_number)
    if not order:
        return False, "Order not found."
    
    if order["refund_status"] == "Refunded":
        return False, "This order has already been fully refunded."
    
    if order["status"] != "Cancelled" and order["status"] != "Delivered":
        return False, "Refund can only be requested for Cancelled or Delivered orders."
    
    order["refund_status"] = "Refunded"
    st.session_state.orders[order["order_number"]] = order
    return True, f"Refund for order {order['order_number']} (Total: ${order['total']}) has been successfully processed."
