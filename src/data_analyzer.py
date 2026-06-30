from __future__ import annotations
import logging
import pandas as pd

# Every function: type hints + docstring + try/except + logger — no exceptions
logger = logging.getLogger(__name__)


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Get summary statistics for all numeric columns in a DataFrame.

    Calculates mean, median, standard deviation, min, and max
    for every numeric column.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        dict: A nested dictionary where keys are column names and values
              are dicts containing 'mean', 'median', 'std', 'min', 'max'.

    Example:
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> get_summary_stats(df)
        {'A': {'mean': 2.0, 'median': 2.0, 'std': 1.0, 'min': 1, 'max': 3},
         'B': {'mean': 5.0, 'median': 5.0, 'std': 1.0, 'min': 4, 'max': 6}}
    """
    try:
        numeric_df = df.select_dtypes(include="number")
        summary_stats = {
            col: {
                "Mean": numeric_df[col].mean(),
                "Median": numeric_df[col].median(),
                "Std": numeric_df[col].std(),
                "Min": numeric_df[col].min(),
                "Max": numeric_df[col].max(),
            }
            for col in numeric_df.columns
        }
        logger.info(f"Successfully calculated summary statistics: {summary_stats}")
        return summary_stats
    except Exception as e:
        logger.error(f"Error calculating the summary statistics: {e}")
        return {}


def find_outliers(df: pd.DataFrame, col: str, threshold: float = 3.0) -> pd.DataFrame:
    """Find outlier rows in a numeric column using the Z-score method.

    Any row whose absolute Z-score exceeds the threshold is flagged as an outlier.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        col (str): The numeric column to check for outliers.
        threshold (float): Z-score cutoff. Default is 3.0 (standard rule).

    Returns:
        pd.DataFrame: Rows from the original DataFrame where the column
                      value is an outlier. Empty DataFrame if none found.

    Example:
        >>> df = pd.DataFrame({'salary': [50000, 52000, 49000, 999999]})
        >>> find_outliers(df, 'salary')
           salary
        3  999999
    """
    try:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0:
            logger.warning(
                f"Standard deviation is zero for column '{col}'."
                f" No outliers can be detected."
            )
            return pd.DataFrame()  # Return empty DataFrame if no variation

        z_scores = (df[col] - mean) / std
        outliers = df[abs(z_scores) > threshold]
        logger.info(
            f"Successfully found outliers in column '{col}': {len(outliers)} rows"
        )
        return outliers

    except KeyError:
        logger.error(f"Column '{col}' not found the in the Dataframe:")
        return pd.DataFrame()
    except TypeError as t:
        logger.error(f"Column '{col}' must be numeric to calculate Z-scores: {t}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error finding outliers in column '{col}': {e}")
        return pd.DataFrame()


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the Pearson correlation matrix for numeric columns.

    Correlation ranges:
    1.0 = perfect positive, 0.0 = no relationship, -1.0 = perfect negative.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        pd.DataFrame: A square correlation matrix, or empty if no numeric columns.
    """
    try:
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty:
            logger.warning("No numeric columns found for correlation matrix ")
            return pd.DataFrame()  # Return empty DataFrame
        corr_matrix = numeric_df.corr(method="pearson")
        logger.info("Successfully computed correalation matrix !")
        return corr_matrix
    except Exception as e:
        logger.error(f"Error computing correlation matrix:{e}")
        return pd.DataFrame()


def count_categories(df: pd.DataFrame, col: str) -> dict[str, int]:
    """count occurrences of each category in a categorical column.
       Useful for understanding the distribution of categorical data.
    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        col (str): The categorical column to count categories for.
    Returns:
        dict[str, int]: A dictionary where keys are unique categories and
                        values are the count of occurrences for each category.
    Example:
        >>> df = pd.DataFrame({'color': ['red', 'blue', 'red', 'green']})
        >>> count_categories(df, 'color')
        {'red': 2, 'blue': 1, 'green': 1}
    """
    try:
        if col not in df.columns:
            logger.error(f"Column '{col}' not found in the DataFrame.")
            return {}

        # value_counts() returns a Series sorted by frequency
        category_counts = {str(k): int(v) for k, v in df[col].value_counts().items()}
        logger.info(
            f"Successfully counted {len(category_counts)} categories in column '{col}'"
        )
        return category_counts

    except Exception as e:
        logger.error(f"Error counting categories in column '{col}': {e}")
        return {}


def flag_missing(df: pd.DataFrame, threshold: float = 0.1) -> list[str]:
    """Flag columns with missing values exceeding a threshold.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        threshold (float): The minimum missing percentage to flag. Defaults to 0.1(10%).

    Returns:
        list[str]: Column names where missing values exceed the threshold.

    Example:
        >>> df = pd.DataFrame({'A': [1, None], 'B': [None, None]})
        >>> flag_missing(df, 50)
        ['B']
    """
    try:
        flagged_columns = [
            col for col in df.columns if df[col].isnull().mean() > threshold
        ]
        logger.info(
            f"Successfully flagged {len(flagged_columns)} columns with missing"
            f"values exceeding {threshold}"
        )
        return flagged_columns
    except Exception as e:
        logger.error(f"Error flagging missing columns: {e}")
        return []


def delivery_time_by_seller(df: pd.DataFrame, min_orders: int = 30) -> pd.DataFrame:
    """Finds sellers with the highest average delivery times.

    Args:
        df: Cleaned DF containing seller_id, delivery_days, order_id, and late_delivery.
        min_orders: Minimum unique orders required to be included (default is 30).

    Returns:
        DataFrame of sellers ranked highest to lowest by average delivery days.
    """
    return (
        df.groupby("seller_id")
        .agg(
            avg_delivery_days=("delivery_days", "mean"),
            order_count=("order_id", "nunique"),
            pct_late=("late_delivery", "mean"),
        )
        .query("order_count >= @min_orders")
        .sort_values("avg_delivery_days", ascending=False)
        .reset_index()
    )


def late_delivery_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the percentage of late deliveries for each month.

    Args:
        df: Cleaned DataFrame containing purchase_month, order_id, and late_delivery.

    Returns:
        DataFrame sorted chronologically by month with order counts & late percentages.
    """
    return (
        df.groupby("purchase_month")
        .agg(
            order_count=("order_id", "nunique"),
            pct_late=("late_delivery", lambda x: x.mean() * 100),
        )
        .sort_values("purchase_month")
        .reset_index()
    )


def worst_sellers_by_on_time_rate(
    df: pd.DataFrame, min_orders: int = 20, top_n: int = 20
) -> pd.DataFrame:
    """Finds sellers with the lowest on-time delivery rates.

    Args:
        df: Cleaned DataFrame containing seller_id, order_id, and late_delivery.
        min_orders: Minimum unique orders required to be included (default is 20).
        top_n: Number of worst-performing sellers to return (default is 20).

    Returns:
        DataFrame of the worst sellers sorted by lowest on-time rate.
    """
    return (
        df.groupby("seller_id")
        .agg(
            order_count=("order_id", "nunique"),
            # late_delivery.mean() is the late rate. 1 minus that is the on-time rate!
            on_time_rate=("late_delivery", lambda x: 1 - x.mean()),
        )
        .query("order_count >= @min_orders")
        .sort_values("on_time_rate", ascending=True)
        .head(top_n)
        .reset_index()
    )


def avg_order_value_by_state(
    df: pd.DataFrame, customers_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculates the average order value and total orders per customer state."""

    logger.info("Calculating average order value. Starting orders: %d", len(df))

    result = (
        df.merge(customers_df, on="customer_id", how="inner")
        .groupby("customer_state")
        .agg(avg_order_value=("price", "mean"), order_count=("order_id", "nunique"))
        .sort_values("avg_order_value", ascending=False)
        .reset_index()
    )

    logger.info("State grouping complete. Outputting %d distinct states.", len(result))

    return result


def add_seller_delivery_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Adds seller-specific delivery context to each individual order.

    Args:
        df: Cleaned orders DataFrame containing seller_id and delivery_days.

    Returns:
        DataFrame with new columns 'seller_avg_delivery' and 'delivery_vs_seller_pct'.
    """
    logger.info("Adding seller delivery ranks to %d orders.", len(df))

    # Create a copy to avoid mutating the original dataframe directly
    result = df.copy()

    # 1. Add the seller's overall average delivery time to every row
    result["seller_avg_delivery"] = result.groupby("seller_id")[
        "delivery_days"
    ].transform("mean")

    # 2.Add the percentile rank(e.g. 0.90 means order was slower than 90% of its others)
    result["delivery_vs_seller_pct"] = result.groupby("seller_id")[
        "delivery_days"
    ].transform(lambda x: x.rank(pct=True))

    logger.info("Successfully added context columns. New shape: %s", result.shape)

    return result


def monthly_order_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes monthly order volume, revenue, and delivery KPIs.

    Args:
        df: The cleaned orders DataFrame containing order items and delivery data.

    Returns:
        DataFrame with monthly aggregated trends and month-over-month growth.
    """
    try:
        # 1. Set and sort the datetime index (Mandatory for time-series)
        df_ts = df.set_index("order_purchase_timestamp").sort_index()

        # 2. Resample to Month-End ("ME") and aggregate core metrics
        monthly = df_ts.resample("ME").agg(
            order_count=("order_id", "nunique"),
            total_revenue=("price", "sum"),
            avg_delivery_days=("delivery_days", "mean"),
            late_pct=("late_delivery", "mean"),
        )

        # 3. Calculate percentage change and rolling averages
        monthly["mom_order_growth"] = monthly["order_count"].pct_change() * 100
        monthly["revenue_4wk_rolling"] = monthly["total_revenue"].rolling(4).mean()

        # 4. Flatten the index and rename the date column for readability
        monthly = monthly.reset_index().rename(
            columns={"order_purchase_timestamp": "month"}
        )

        logger.info("Monthly trends computed: %d months", len(monthly))
        return monthly

    except KeyError as e:
        logger.exception("Missing column for time-series: %s", e)
        raise


def weekly_volume_with_rolling_avg(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """
    Calculates weekly order volume and its moving average.

    Args:
        df: The cleaned orders DataFrame.
        window: The number of weeks to include in the rolling average.

    Returns:
        DataFrame with columns: week, order_count, and rolling_avg.
    """
    df_ts = df.set_index("order_purchase_timestamp").sort_index()
    weekly = df_ts.resample("W").agg(order_count=("order_id", "nunique"))

    # Apply the rolling window
    weekly["rolling_avg"] = weekly["order_count"].rolling(window=window).mean()

    # Check and log expected NaN values
    if weekly["rolling_avg"].isna().any():
        logger.warning(
            "NaN values present in rolling average (expected for first %d rows).",
            window - 1,
        )

    # Flatten the index and rename the date column
    weekly = weekly.reset_index().rename(columns={"order_purchase_timestamp": "week"})

    return weekly


def detect_growth_periods(
    df: pd.DataFrame, metric: str = "order_count"
) -> pd.DataFrame:
    """
    Classifies each month as growth, decline, or stable based on mom_order_growth.

    Args:
        df: The cleaned orders DataFrame.
        metric: The column to base the growth classification on.

    Returns:
        DataFrame with an added 'period_label' column.
    """
    # Call the previous function internally
    monthly = monthly_order_trends(df)

    # Define the classification logic
    def classify_growth(pct):
        if pd.isna(pct):
            return "unknown"
        elif pct > 5.0:
            return "growth"
        elif pct < -5.0:
            return "decline"
        else:
            return "stable"

    # Apply the logic to create the new column
    monthly["period_label"] = monthly["mom_order_growth"].apply(classify_growth)

    logger.info("Growth periods classified for %d months.", len(monthly))
    return monthly


def peak_hour_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes order volume by hour of the day to identify peak shopping times.

    Args:
        df(pd.DataFrame):Cleaned orders  containing 'purchase_hour' & 'order_id'.

    Returns:
        pd.DataFrame:Grouped DataFrame sorted by hour with 'order_count' &'pct_of_total'.
    """
    # 1. Group by hour and count unique orders
    hourly = (
        df.groupby("purchase_hour")
        .agg(order_count=("order_id", "nunique"))
        .reset_index()
        .sort_values("purchase_hour")
    )

    # 2. Calculate percentage of total daily orders
    total_orders = hourly["order_count"].sum()
    hourly["pct_of_total"] = (hourly["order_count"] / total_orders) * 100

    # 3. Identify and log the peak hour
    peak_row = hourly.loc[hourly["order_count"].idxmax()]
    logger.info(
        "Peak hour is %02d:00 with %.1f%% of total orders.",
        int(peak_row["purchase_hour"]),  # type: ignore
        peak_row["pct_of_total"],
    )

    return hourly


# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Short example to test if the file runs correctly
    test_df = pd.DataFrame({"A": [10, 20, 30], "B": ["cat", "dog", "cat"]})
    print(get_summary_stats(test_df))
    print(count_categories(test_df, "B"))
