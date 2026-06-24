import pandas as pd
import logging
import chardet
from pathlib import Path
from typing import Generator
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


def load_csv(path: str) -> pd.DataFrame | None:
    """Load a CSV file into a pandas DataFrame.

    Args:
        path (str): The file path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(path)
        logger.info(f"Successfully loaded CSV file from {path}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file at {path}: {e}")
        return None


def get_shape(df: pd.DataFrame) -> tuple[int, int] | None:
    """Get the shape of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to get the shape of.

    Returns:
        tuple[int, int]: A tuple containing the number of rows and columns.
    """
    try:
        shape = df.shape
        logger.info(f"Successfully got shape of DataFrame: {shape}")
        return shape
    except AttributeError as e:
        logger.error(f"Error getting shape from DataFrame: {e}")
        return None


def get_columns(df: pd.DataFrame) -> list[str] | None:
    """Get the column names of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to get the column names of.

    Returns:
        list[str]: A list of column names in the DataFrame.
    """
    try:
        # ✅ Task 1 — list comprehension (replaces .tolist())
        columns = [col for col in df.columns]
        logger.info(f"Successfully got column names from DataFrame: {columns}")
        return columns
    except AttributeError as e:
        logger.error(f"Error getting column names from DataFrame: {e}")
        return None


def filter_by_value(df: pd.DataFrame, col: str, val: float) -> pd.DataFrame | None:
    """Filter the DataFrame by a specific value in a column.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        col (str): The column name to filter by.
        val (float): The value to filter by.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only rows where the
        specified column has the specified value.


    """
    try:
        filtered_df = df[df[col] == val]
        logger.info(
            f"Successfully filtered DataFrame by column '{col}' and value {val}"
        )
        return filtered_df
    except Exception as e:
        logger.error(
            f"Error filtering DataFrame by column '{col}' and value {val}: {e}"
        )
        return None


def get_missing_counts(df: pd.DataFrame) -> dict[str, int] | None:
    """Get the count of missing values in each column of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        dict[str, int]: A dictionary where keys are column names and
        values are the count of missing values in each column.

    Example:
        >>> df = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, None]})
        >>> get_missing_counts(df)
        {'A': 1, 'B': 1}
    """
    try:
        missing_counts = df.isnull().sum().to_dict()
        logger.info(f"Successfully got missing counts from DataFrame: {missing_counts}")
        return missing_counts
    except AttributeError as e:
        logger.error(f"Error getting missing counts from DataFrame: {e}")
        return None


def get_missing_percent(df: pd.DataFrame) -> dict[str, float] | None:
    """Get the missing value percentage for each column.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        dict[str, float]: A dictionary where keys are column names and
        values are the percentage of missing values (0-100).

    Example:
        >>> df = pd.DataFrame({'A': [1, None], 'B': [None, None]})
        >>> get_missing_percent(df)
        {'A': 50.0, 'B': 100.0}
    """
    try:
        # ✅ Task 2 — dict comprehension
        missing_percent = {col: df[col].isnull().mean() * 100 for col in df.columns}
        logger.info(f"Successfully calculated missing percentages: {missing_percent}")
        return missing_percent
    except AttributeError as e:
        logger.error(f"Error calculating missing percentages: {e}")
        return None


def get_high_missing_columns(
    df: pd.DataFrame, threshold: float = 5.0
) -> list[str] | None:
    """Get column names where missing values exceed a threshold percentage.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        threshold (float): The minimum missing percentage to flag. Defaults to 5.0.

    Returns:
        list[str]: Column names where missing values exceed the threshold.

    Example:
        >>> df = pd.DataFrame({'A': [1, None, None], 'B': [1, 2, 3]})
        >>> get_high_missing_columns(df, threshold=5.0)
        ['A']
    """
    try:
        missing_percent = get_missing_percent(df)
        # ✅ Task 3 — filter comprehension
        high_missing = [
            col for col in missing_percent if missing_percent[col] > threshold
        ]
        logger.info(f"Columns with >{threshold}% missing: {high_missing}")
        return high_missing
    except TypeError as e:
        logger.error(f"Error filtering high missing columns: {e}")
        return None


def chunk_csv_reader(filepath: str, chunk_size: int = 1000) -> Generator:
    """Read a CSV file in chunks to handle large files efficiently.

    Args:
        filepath (str): The path to the CSV file.
        chunk_size (int): The number of rows per chunk. Defaults to 1000.

    Yields:
        pd.DataFrame: A chunk of the DataFrame read from the CSV file.

    Example:
        for chunk in chunk_csv_reader('large_data.csv', chunk_size=500):
            print(chunk.shape)
    """
    try:
        for chunk in pd.read_csv(filepath, chunksize=chunk_size):
            logger.info(f"Successfully read a chunk of size {chunk.shape}")
            yield chunk
    except Exception as e:
        logger.error(f"Error reading CSV file in chunks from {filepath}: {e}")
        return


def load_csv_safe(
    path: str,
    encoding: str | None = None,
    dtype_spec: dict[str, str] | None = None,
    low_memory: bool = False,
) -> pd.DataFrame:
    """Load a CSV file with automatic encoding detection and dtype control.

    Args:
        path: Absolute or relative path to the CSV file.
        encoding: Character encoding (e.g. 'utf-8', 'latin-1'). If None,
            chardet auto-detects it from the first 50 000 bytes.
        dtype_spec: Optional dict mapping column names to dtype strings,
            e.g. {"price": "float64", "customer_id": "str"}.
        low_memory: Pass True for files larger than ~100 MB to avoid
            dtype guessing across the entire file.

    Returns:
        pd.DataFrame: Loaded DataFrame with applied dtypes.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file is empty or unreadable.

    Example:
        >>> df = load_csv_safe("data/raw/olist/olist_orders_dataset.csv")
        >>> print(df.shape)
        (99441, 8)
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        if encoding is None:
            raw_bytes = file_path.read_bytes()[:50_000]
            detected = chardet.detect(raw_bytes)
            encoding = detected.get("encoding", "utf-8") or "utf-8"
            logger.info(
                "Auto-detected encoding: %s (confidence: %.2f)",
                encoding,
                detected.get("confidence", 0),
            )

        df = pd.read_csv(
            file_path,
            encoding=encoding,
            dtype=dtype_spec,
            low_memory=low_memory,
        )

        if df.empty:
            raise ValueError(f"CSV loaded as empty DataFrame: {path}")

        logger.info(
            "Loaded %s — shape: %s, encoding: %s", file_path.name, df.shape, encoding
        )
        return df

    except FileNotFoundError:
        logger.error("File not found: %s", path)
        raise
    except UnicodeDecodeError as e:
        logger.error("Encoding error in %s: %s. Try encoding='latin-1'", path, e)
        raise
    except pd.errors.ParserError as e:
        logger.error("CSV parse error in %s: %s", path, e)
        raise
    except pd.errors.EmptyDataError:
        logger.error("File is completely empty: %s", path)
        raise ValueError(f"CSV is empty: {path}")
    except ValueError:
        logger.exception("Data validation failed for file: %s", path)
        raise


def load_multiple_csvs(
    directory: str, pattern: str = "*.csv"
) -> dict[str, pd.DataFrame]:
    """Load all matching CSV files from a directory into a dictionary of DataFrames.

    Args:
        directory (str): Path to the folder containing the CSV files.
        pattern (str): Glob pattern to filter files. Defaults to "*.csv".

    Returns:
        dict[str, pd.DataFrame]: A dictionary mapping the filename (without extension)
        to its corresponding pandas DataFrame.

    Raises:
        FileNotFoundError: If the provided directory does not exist or is not a folder.

    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    Data_dic = {}
    failed_files = []
    # what is .glob()
    # .glob() returns a generator of Path objects matching the pattern
    for file_path in dir_path.glob(pattern):
        try:
            df = load_csv_safe(str(file_path))
            Data_dic[file_path.stem] = df
        except (
            FileNotFoundError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ):
            logger.exception("Failed to load %s", file_path.name)
            failed_files.append(file_path.name)
            # .stem returns the filename without the extension
    if not Data_dic:
        logger.warning("No CSV files loaded from %s (pattern=%s)", directory, pattern)
    else:
        logger.info(
            "Successfully loaded %s files from %s (%s failed)",
            len(Data_dic),
            directory,
            len(failed_files),
        )
    return Data_dic


def infer_dtypes_report(df: pd.DataFrame) -> dict[str, str]:
    """Inspect column dtypes and warn about unparsed datetime columns.

    Args:
        df (pd.DataFrame): The pandas DataFrame to inspect.

    Returns:
        dict[str, str]: A dictionary mapping column names to their string data types.

    Raises:
        TypeError: If the provided 'df' input is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("Expected a pandas DataFrame, got %s", type(df))
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)}")
    res_Dict = {}
    for col in df.columns:
        d_type = str(df[col].dtype)
        res_Dict[col] = d_type

        is_datetime_like = any(
            keyword in col.lower() for keyword in ("date", "time", "timestamp")
        )
        if d_type == "object" and is_datetime_like:
            logger.warning(
                "Column '%s' looks like a datetime but was loaded as "
                "object — consider pd.to_datetime()",
                col,
            )

    return res_Dict


def delivery_time_by_seller(df: pd.DataFrame, min_orders: int = 30) -> pd.DataFrame:
    """Finds sellers with the highest average delivery times.

    Args:
        df: Cleaned DataFrame containing seller_id, delivery_days, order_id, and late_delivery.
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
        DataFrame sorted chronologically by month with order counts and late percentages.
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

    # 2. Add the percentile rank (e.g., 0.90 means this order was slower than 90% of their others)
    result["delivery_vs_seller_pct"] = result.groupby("seller_id")[
        "delivery_days"
    ].transform(lambda x: x.rank(pct=True))

    logger.info("Successfully added context columns. New shape: %s", result.shape)

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test 1: single file with encoding auto-detect
    SCRIPT_DIR = Path(__file__).resolve().parent
    TARGET_FILE = (
        SCRIPT_DIR.parent / "data" / "raw" / "olist" / "olist_orders_dataset.csv"
    )

    orders = load_csv_safe(TARGET_FILE)
    print(orders.shape)

    # Test 2: multi-file load
    all_dfs = load_multiple_csvs("data/raw/olist/")
    print(list(all_dfs.keys()))

    # Test 3: dtype report
    report = infer_dtypes_report(orders)
    print(report)
