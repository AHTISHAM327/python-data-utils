import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.
    Args:
        path (str): The file path to the CSV file.
    Returns:
        pd.DataFrame: The loaded DataFrame."""
    return pd.read_csv(path)


def get_shape(df: pd.DataFrame) -> tuple[int, int]:
    """Get the shape of a DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to get the shape of.
    Returns:
        tuple[int, int]: A tuple containing the number of rows and columns
        in the DataFrame.
    """
    return df.shape


def get_columns(df: pd.DataFrame) -> list[str]:
    """Get the column names of a DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to get the column names of.
    Returns:
        list[str]: A list of column names in the DataFrame."""
    return df.columns.tolist()


def filter_by_value(df: pd.DataFrame, col: str, val: float) -> pd.DataFrame:
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
    return df[df[col] == val]


def get_missing_counts(df: pd.DataFrame) -> dict[str, int]:
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
    return df.isnull().sum().to_dict()
