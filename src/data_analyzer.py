import pandas as pd
import logging

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


# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Short example to test if the file runs correctly
    test_df = pd.DataFrame({"A": [10, 20, 30], "B": ["cat", "dog", "cat"]})
    print(get_summary_stats(test_df))
    print(count_categories(test_df, "B"))
