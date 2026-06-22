"""
End-to-end cleaning pipeline test for the Olist Orders dataset.
Run with: python tests/test_cleaning_pipeline.py
"""

import logging
import sys

sys.path.insert(0, ".")

from src.data_loader import load_csv_safe
from src.data_cleaner import (
    audit_nulls,
    fill_nulls_by_strategy,
    convert_datetime_columns,
    optimize_memory,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# 1. Load
orders = load_csv_safe("data/raw/olist/olist_orders_dataset.csv")
print(f"\n✓ Loaded: {orders.shape}")

# 2. Audit nulls
null_report = audit_nulls(orders)
print(f"\n✓ Null audit:\n{null_report.to_string()}")

# 3. Convert dates
date_cols = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
orders = convert_datetime_columns(orders, date_cols)
print(f"\n✓ Datetime dtypes confirmed:\n{orders[date_cols].dtypes}")

# 4. Optimize memory
orders_opt, log = optimize_memory(orders)
print(f"\n✓ Optimized columns: {log}")

print("\n── FINAL DATASET ──")
print(f"Shape:   {orders_opt.shape}")
print(f"Dtypes:\n{orders_opt.dtypes}")
print(f"\nFirst 3 rows:\n{orders_opt.head(3)}")
