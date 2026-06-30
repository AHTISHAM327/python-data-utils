"""
EDA Pipeline — orchestrates loading, cleaning, and feature engineering
using method chaining. Entry point: run_full_pipeline().
"""

import sys
import pandas as pd
import logging

sys.path.insert(0, ".")
from src.data_loader import load_multiple_csvs
from src.data_cleaner import (
    audit_nulls,
    convert_datetime_columns,
    optimize_memory,
)

logger = logging.getLogger(__name__)


def _coerce_dataframe(result):
    """Return a DataFrame whether optimize_memory returns one directly
    or as the first element of a (DataFrame, stats) tuple.

    Verify your actual Day 2 optimize_memory return signature — this
    helper keeps the pipeline working either way.
    """
    return result[0] if isinstance(result, tuple) else result


def engineer_order_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds calculated columns for delivery days, late status, month, and hour.

    Args:
        df (pd.DataFrame): The cleaned orders DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame containing the engineered feature columns.
    """
    df_engineered = df.assign(
        delivery_days=lambda x: (
            x["order_delivered_customer_date"] - x["order_purchase_timestamp"]
        ).dt.days,
        late_delivery=lambda x: x["order_delivered_customer_date"]
        > x["order_estimated_delivery_date"],
        purchase_month=lambda x: x["order_purchase_timestamp"].dt.to_period("M"),
        purchase_hour=lambda x: x["order_purchase_timestamp"].dt.hour,
    )

    total_orders = len(df_engineered)
    late_count = df_engineered["late_delivery"].sum()
    late_pct = (late_count / total_orders) * 100 if total_orders > 0 else 0.0

    logger.info(
        "Late deliveries: %d of %d (%.1f%%)", late_count, total_orders, late_pct
    )
    return df_engineered


def merge_order_items(orders: pd.DataFrame, order_items: pd.DataFrame) -> pd.DataFrame:
    """Merges orders with their individual items on order_id using a LEFT JOIN.
    Args:
        orders (pd.DataFrame): The orders DataFrame.
        order_items (pd.DataFrame): The order items DataFrame.

    Returns:
        pd.DataFrame: A merged DataFrame at the order-item granularity.
    """
    logger.info("Shape before merge: %s", orders.shape)
    merged_df = orders.merge(order_items, on="order_id", how="left")
    logger.info("Shape after merge: %s", merged_df.shape)
    return merged_df


def filter_delivered_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Filters for delivered orders with valid delivery times (1 to 119 days)."""
    original_count = len(df)
    filtered_df = df.query(
        "order_status == 'delivered' and delivery_days > 0 and delivery_days < 120"
    )
    removed_count = original_count - len(filtered_df)

    logger.info("Filtered out %d invalid or undelivered rows.", removed_count)
    return filtered_df


def run_full_pipeline(data_dir: str) -> pd.DataFrame:
    """Load, clean, and feature-engineer Olist orders end-to-end.

    Args:
        data_dir: Path to the directory containing Olist CSV files.

    Returns:
        pd.DataFrame: Cleaned, feature-engineered orders DataFrame
            ready for groupby analysis and visualisation.
    """
    try:
        logger.info("Starting full EDA pipeline on: %s", data_dir)
        all_dfs = load_multiple_csvs(data_dir)

        null_report = audit_nulls(all_dfs["olist_orders_dataset"])
        logger.info("Null audit on orders table:\n%s", null_report)

        date_cols = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]

        # 1. Extract the dataset into a variable
        orders_df = all_dfs["olist_orders_dataset"]

        assert orders_df is not None, "Data cannot be None before pipeline"

        # 3. The core method chain (now using our safe variable)
        result = (
            orders_df.pipe(convert_datetime_columns, columns=date_cols)
            .pipe(lambda df: _coerce_dataframe(optimize_memory(df)))
            .pipe(engineer_order_features)
            .pipe(
                lambda df: merge_order_items(df, all_dfs["olist_order_items_dataset"])
            )
            .pipe(filter_delivered_orders)  # type: ignore
        )

        logger.info("Pipeline complete. Final shape: %s", result.shape)
        return result

    except KeyError as e:
        logger.exception("Missing expected table or column: %s", e)
        raise
    except Exception:
        logger.exception("EDA pipeline failed for data_dir=%s", data_dir)
        raise


if __name__ == "__main__":

    # Configure logging to show up in the terminal
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Run the pipeline
    data_path = "data/raw/olist/"
    df_final = run_full_pipeline(data_path)

    print(f"\nSuccess! Final DataFrame shape: {df_final.shape}")
