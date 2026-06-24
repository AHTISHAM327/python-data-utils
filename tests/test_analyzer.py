import sys
import pandas as pd
import logging

# Ensure Python can find the src folder from the root directory
sys.path.insert(0, ".")

from src.eda_pipeline import run_full_pipeline
from src.data_loader import (
    delivery_time_by_seller,
    late_delivery_by_month,
    worst_sellers_by_on_time_rate,
    avg_order_value_by_state,
    add_seller_delivery_rank,
)


def run_tests():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n--- Loading Pipeline ---")
    # Adjust path if your raw data is stored elsewhere
    df = run_full_pipeline("data/raw/olist/")

    print("\n--- Loading Customers Table ---")
    customers_df = pd.read_csv("data/raw/olist/olist_customers_dataset.csv")

    print("\n--- Q1: Delivery Time by Seller ---")
    q1 = delivery_time_by_seller(df)
    print(q1.head(3))

    print("\n--- Q2: Late Delivery by Month ---")
    q2 = late_delivery_by_month(df)
    print(q2.head(3))

    print("\n--- Q3: Worst Sellers by On-Time Rate ---")
    q3 = worst_sellers_by_on_time_rate(df)
    print(q3.head(3))

    print("\n--- Q4: Avg Order Value by State ---")
    q4 = avg_order_value_by_state(df, customers_df)
    print(q4.head(3))

    print("\n--- Q5: Seller Delivery Rank ---")
    q5 = add_seller_delivery_rank(df)
    # Sub-setting columns for a cleaner print output
    print(
        q5[
            [
                "order_id",
                "seller_id",
                "delivery_days",
                "seller_avg_delivery",
                "delivery_vs_seller_pct",
            ]
        ].head(3)
    )


if __name__ == "__main__":
    run_tests()
