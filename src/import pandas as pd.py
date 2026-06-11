import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_csv(path: str) -> pd.DataFrame | None:
    """Load a CSV file into a pandas DataFrame.
    Args:
        path (str): The file path to the CSV file.
    Returns:
        pd.DataFrame: The loaded DataFrame."""
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
        tuple[int, int]: A tuple containing the number of rows and columns
        in the DataFrame.
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
        list[str]: A list of column names in the DataFrame."""
    try:
        columns = df.columns.tolist()
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
        logger.info(f"Successfully filtered DataFrame by column {col} and value {val}")
        return filtered_df
    except Exception as e:
        logger.error(f"Error filtering DataFrame by column {col} and value {val}: {e}")
        return None


def get_missing_counts(df: pd.DataFrame) -> dict[str, int] | None:
    """Get the count of missing values in each column of a DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to analyze.
    Returns:
        dict[str, int]: A dictionary where the keys are column names and the
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
