import pandas as pd
import logging
from typing import Literal

logger = logging.getLogger(__name__)


def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any null values from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: A new DataFrame with rows containing null values removed.
    """
    try:
        clean_df = df.dropna()
        logger.info(
            f"Succesfully dropped null values. Original shape:{df.shape},"
            f"New Shape:{clean_df.shape}"
        )
        return clean_df
    except Exception as e:
        logger.error(f"Error dropping null values:{e}")
        return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to clean.
    Returns:
        pd.DataFrame: A new DataFrame with duplicate rows removed.
    """
    try:
        clean_df = df.drop_duplicates()
        logger.info(
            f"Successfully removed the duplicates.Original shape:{df.shape}"
            f", new shape : {clean_df.shape}"
        )
        return clean_df
    except Exception as e:
        logger.error(f"Error removing duplicates: {e}")
        return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all column names to lowercase and replace spaces with underscores.
    Args:
        df(pd.DataFrame): The DataFrame with original column names.
    Returns:
        pd.DataFrame: The DataFrame with cleaned column names.
    Example:
        df = pd.DataFrame({"Column A": [1, 2], "Column B": [3, 4]})
        cleaned_df = clean_column_names(df)
        print(cleaned_df.columns)#Output: Index(['column_a', 'column_b'],dtype='object')
    """
    try:
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        logger.info(f"Column names cleaned successfully: {df.columns.tolist()}")
        return df
    except Exception as e:
        logger.error(f"An error occurred while cleaning column names: {e}")
        return df


def audit_nulls(df: pd.DataFrame) -> pd.DataFrame | None:
    """Analyzes a DataFrame for missing values and suggests mitigation strategies.

    Args:
        df (pd.DataFrame): The pandas DataFrame to audit.

    Returns:
        pd.DataFrame | None: A report DataFrame containing columns with missing values.
            Returns an empty DataFrame with the correct columns if no nulls are found.
            Returns None if an error occurs.

    Raises:
        AttributeError: If the input object lacks pandas DataFrame methods.
        TypeError: If the input does not support required operations like len().
        ZeroDivisionError: If DataFrame has 0 rows, preventing percentage calculation.
    """
    try:
        null_counts = df.isnull().sum()
        null_counts = null_counts[null_counts > 0]

        if null_counts.empty:
            logging.info("Found 0 columns with nulls")
            return pd.DataFrame(
                columns=["column", "null_count", "null_pct", "suggested_strategy"]
            )
        total_rows = len(df)
        audit_data = []

        for col, count in null_counts.items():
            pct = round((count / total_rows) * 100, 2)

            if pct < 5.0:
                strategy = "drop rows"
            elif pct < 40.0:
                strategy = "fill with median/mode"
            else:
                strategy = "investigate or drop column"

            audit_data.append(
                {
                    "column": col,
                    "null_count": count,
                    "null_pct": pct,
                    "suggested_strategy": strategy,
                }
            )
        report_df = pd.DataFrame(audit_data)
        logger.info("Found %d columns with nulls", len(report_df))
        return report_df

    except AttributeError as e:
        logger.error("Input error: Object is missing pandas DataFrame methods. %s", e)
        return None
    except TypeError as e:
        logger.error("Type error: Input does not support required operations. %s", e)
        return None
    except ZeroDivisionError as e:
        logger.error(
            "Math error: DataFrame has 0 rows, preventing percentage calculation. %s", e
        )
        return None


def fill_nulls_by_strategy(
    df: pd.DataFrame,
    strategy: dict[str, Literal["median", "mean", "mode", "drop", "zero", "ffill"]],
) -> pd.DataFrame | None:
    """Fills or drops missing values based on a column-specific strategy.

    Args:
        df (pd.DataFrame): The input pandas DataFrame.
        strategy (dict): A dictionary mapping column names to a specific string
            strategy ('median', 'mode', 'drop', 'zero', 'ffill').

    Returns:
        pd.DataFrame | None: A new, modified DataFrame with strategies applied.
            Returns None if a critical error occurs.

    Raises:
        AttributeError: If 'df' lacks pandas methods (e.g., .copy()).
        KeyError: If a specified column in the strategy dict is missing from  DataFrame.
        IndexError: If calculating 'mode' fails on an entirely empty/null column.
    """
    try:
        clean_df = df.copy()

        for col, strat in strategy.items():
            if col not in clean_df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

            if strat == "drop":
                clean_df = clean_df.dropna(subset=[col])
            elif strat == "median":
                clean_df[col] = clean_df[col].fillna(clean_df[col].median())
            elif strat == "mode":
                clean_df[col] = clean_df[col].fillna(clean_df[col].mode()[0])
            elif strat == "zero":
                clean_df[col] = clean_df[col].fillna(0)
            elif strat == "ffill":
                clean_df[col] = clean_df[col].ffill()
            elif strat == "mean":
                clean_df[col] = clean_df[col].fillna(clean_df[col].mean())

            else:
                logger.warning(
                    "Unknown strategy '%s' for column '%s'. Skipping.", strat, col
                )

        logger.info("Successfully applied strategies to %d columns", len(strategy))
        return clean_df

    except AttributeError as e:
        logger.error("Input error: Object is missing pandas DataFrame methods. %s", e)
        return None
    except KeyError as e:
        logger.error("Column error: %s", e)
        return None
    except IndexError as e:
        logger.error("Data error: Cannot calculate mode on empty column. %s", e)
        return None


def convert_datetime_columns(
    df: pd.DataFrame,
    columns: list[str],
    format: str | None = None,
    errors: Literal["raise", "coerce", "ignore"] = "coerce",
) -> pd.DataFrame | None:
    """Convert specified columns to datetime format.

    Args:
        df (pd.DataFrame): The pandas DataFrame containing the columns to convert.
        columns (list[str]): A list of column names to convert to datetime format.
        format (str | None): The strftime format string for parsing.
            If None, pandas will attempt to infer the format automatically.
        errors (Literal["raise", "coerce", "ignore"]): How to handle parsing errors.
            'raise' will raise an exception on invalid parsing.
            'coerce' will set invalid values to NaT.
            'ignore' will leave the column unchanged on error.

    Returns:
        pd.DataFrame: The DataFrame with converted datetime columns.
        None: If an error occurs during conversion.

    Raises:
        KeyError: If any specified column is not found in the DataFrame.
        TypeError: If a column contains data that cannot be converted.
    """
    try:
        for col in columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
            df[col] = pd.to_datetime(df[col], format=format, errors=errors)
            logger.info("Converted column '%s' to datetime", col)
        return df
    except KeyError as e:
        logger.error("Column error: %s", e)
        return None
    except TypeError as e:
        logger.error("Type error: %s", e)
        return None


def optimize_memory(
    df: pd.DataFrame,
    max_category_cardinality: int = 50,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Downcast columns to more memory-efficient dtypes.

    Converts object columns with low cardinality to the category dtype
    and downcasts int64 columns to int32 when values fit within range.

    Args:
        df (pd.DataFrame): The DataFrame to optimize.
        max_category_cardinality (int): Maximum number of unique values
            for an object column to be converted to category.
            Defaults to 50.

    Returns:
        tuple[pd.DataFrame, dict[str, str]]: A tuple containing the
            new_df DataFrame and a log of conversions performed.
            The log maps column names to their dtype changes, e.g.
            {"order_status": "object → category"}.

    Raises:
        TypeError: If the input is not a pandas DataFrame or a column
            dtype is incompatible with the intended conversion.
        ValueError: If max_category_cardinality is non-integer or negative.
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(df)}")
        if (
            not isinstance(max_category_cardinality, int)
            or max_category_cardinality < 0
        ):
            raise ValueError(
                f"max_category_cardinality must be a non-negative integer, "
                f"got {max_category_cardinality}"
            )

        before_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
        new_df = df.copy()
        conversion_log: dict[str, str] = {}

        for col in new_df.select_dtypes(include="object").columns:
            unique_count = new_df[col].nunique()
            if unique_count > 0 and unique_count < max_category_cardinality:
                new_df[col] = new_df[col].astype("category")
                conversion_log[col] = "object → category"

        for col in new_df.select_dtypes(include=["int64"]).columns:
            col_max = new_df[col].max()
            if pd.notna(col_max) and col_max < 2_147_483_647:
                new_df[col] = new_df[col].astype("int32")
                conversion_log[col] = "int64 → int32"

        after_mem = new_df.memory_usage(deep=True).sum() / (1024 * 1024)
        reduction = (
            ((before_mem - after_mem) / before_mem * 100) if before_mem > 0 else 0.0
        )
        logger.info(
            "Memory: %.2f MB -> %.2f MB (%.1f%% reduction)",
            before_mem,
            after_mem,
            reduction,
        )
        return new_df, conversion_log

    except TypeError as e:
        logger.error("Type error during memory optimization: %s", e)
        return df, {}
    except ValueError as e:
        logger.error("Value error during memory optimization: %s", e)
        return df, {}
    except Exception as e:
        logger.error("Unexpected error during memory optimization: %s", e)
        return df, {}


if __name__ == "__main__":
    logging.basicConfig(
        filename="data_cleaner.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.info("Data cleaner module executed directly.")

    import sys

    sys.path.insert(0, ".")
    from src.data_loader import load_csv_safe

    orders = load_csv_safe("data/raw/olist/olist_orders_dataset.csv")

    # --- 2. Test fill_nulls_by_strategy on Products ---
    print("\n--- Testing fill_nulls_by_strategy on Products ---")
    products = load_csv_safe("data/raw/olist/olist_products_dataset.csv")

    # Define the strict strategy dictionary
    test_strategy = {
        "product_category_name": "mode",  # Text -> fill with most common category
        "product_weight_g": "median",  # Numeric -> fill with middle weight
        "product_photos_qty": "drop",  # Cannot guess photo count -> drop the row
    }

    # Run the execution function
    cleaned_products = fill_nulls_by_strategy(products, test_strategy)

    # Print the proof
    if cleaned_products is not None:
        print(f"\nOriginal Products Shape: {products.shape}")
        print(f"Cleaned Products Shape:  {cleaned_products.shape}")

        print("\nChecking remaining nulls in our targeted columns:")
        print(
            cleaned_products[
                ["product_category_name", "product_weight_g", "product_photos_qty"]
            ]
            .isnull()
            .sum()
        )
        date_cols = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
        orders_clean = convert_datetime_columns(orders, date_cols)
        print(orders_clean.dtypes)
        print(orders_clean[date_cols].head(3))

        optimized_orders, log = optimize_memory(orders_clean)
        print("Conversions:", log)
        before_mb = orders_clean.memory_usage(deep=True).sum() / 1e6
        after_mb = optimized_orders.memory_usage(deep=True).sum() / 1e6
        reduction = ((before_mb - after_mb) / before_mb * 100) if before_mb > 0 else 0.0
        print(f"Memory: {before_mb:.2f} MB → {after_mb:.2f} MB ({reduction:.1f}%)")
