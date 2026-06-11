import pandas as pd
import logging

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

    Example:
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> filter_by_value(df, 'A', 2)
               A  B
            1  2  5
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Sample DataFrame for testing
    df = pd.DataFrame(
        {
            "age": [25, None, 30, None, 22],
            "salary": [50000, 60000, None, None, 45000],
            "name": ["Ali", "Bob", None, "Dan", "Eve"],
        }
    )

    print("\n--- get_columns ---")
    print(get_columns(df))

    print("\n--- get_shape ---")
    print(get_shape(df))

    print("\n--- get_missing_counts ---")
    print(get_missing_counts(df))

    print("\n--- get_missing_percent ---")
    print(get_missing_percent(df))

    print("\n--- get_high_missing_columns (threshold=5%) ---")
    print(get_high_missing_columns(df, threshold=5.0))

    print("\n--- filter_by_value (age == 25) ---")
    print(filter_by_value(df, "age", 25))
