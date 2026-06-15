import pandas as pd
import logging

logger = logging.getLogger(__name__)


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


def generate_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a quality report for the DataFrame,
    including missing values and data types.
    Args:
        df(pd.DataFrame): The DataFrame to analyze.
    Returns:
        pd.DataFrame: A DataFrame containing the quality report.
    Example:
        df = pd.DataFrame({"A": [1, None], "B": ["x", "y"]})
        report = generate_quality_report(df)
        print(report)
    """
    try:
        report = pd.DataFrame(
            {
                "column_name": df.columns,
                "data_type": df.dtypes,
                "missing_values": df.isnull().sum(),
            }
        )
        logger.info("Quality report generated successfully.")
        return report
    except Exception as e:
        logger.error(f"An error occurred while generating the quality report: {e}")
        return pd.DataFrame()


def safe_load_csv(file_path: str) -> pd.DataFrame:
    """Safely load a CSV file into a DataFrame, handling potential errors.
    Args:
        file_path (str): The path to the CSV file.
    Returns:
        pd.DataFrame: The loaded DataFrame, or an empty DataFrame if an error occurs
    Example:
        df = safe_load_csv("data.csv")
        if df.empty:
            logger.warning("The DataFrame is empty. Check the file path and contents.")
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"CSV file loaded successfully: {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"No file found at path: {file_path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        logger.error(f"The file at path: {file_path} is empty.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"An error occurred while loading the CSV file: {e}")
        return pd.DataFrame()


def profile_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    """Profile numeric columns in the DataFrame, providing summary statistics.
    Args:
        df(pd.DataFrame): The DataFrame to profile.
    Returns:
        pd.DataFrame: A DataFrame containing summary statistics for numeric columns.
    Example:
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        profile = profile_numeric_data(df)
        print(profile)
    """
    try:
        numeric_df = df.select_dtypes(include="number")
        profile = numeric_df.describe().transpose()
        logger.info("Numeric data profiled successfully.")
        return profile
    except Exception as e:
        logger.error(f"An error occurred while profiling numeric data: {e}")
        return pd.DataFrame()


def flatten_dictionary(nes_dict: dict) -> dict:
    """Flatten a nested dictionary into a single-level dictionary with compound keys.
    Args:
        nes_dict (dict): The nested dictionary to flatten.
    Returns:
        dict: A flattened dictionary where nested keys are combined into a single key.
    Example:
        nested_dict = {'A': {'mean': 1, 'std': 0.5}, 'B': {'mean': 2, 'std': 0.3}}
        flat_dict = flatten_dictionary(nested_dict)
        print(flat_dict)  # Output: {'A_mean': 1, 'A_std': 0.5, 'B_mean':2,'B_std':0.3}
    """
    try:
        flat_dict = {}
        for key, subdict in nes_dict.items():
            for subkey, value in subdict.items():
                flat_key = f"{key}_{subkey}"
                flat_dict[flat_key] = value
        logger.info("Dictionary flattened successfully.")
        return flat_dict
    except Exception as e:
        logger.error(f"An error occurred while flattening the dictionary: {e}")
        return {}


def validate_dataframe(df: pd.DataFrame, required_columns: list = None) -> bool:
    """Validate that the DataFrame contains required columns and has no missing values.
    Args:
        df (pd.DataFrame): The DataFrame to validate.
        required_columns (list, None ): A list of column names that must be present
        in the DataFrame.
    Returns:
        bool: True if the DataFrame is valid, False otherwise.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        for col in missing_columns:
            logger.error(f"Required column '{col}' is missing from the DataFrame.")
            return False

    if df.isnull().sum().sum() > 0:
        logger.error("The DataFrame contains missing values.")
        return False

    logger.info("DataFrame validation passed.")
    return True


# p07
def count_total_rows(file_path: str, chunk_size: int = 1000) -> int:
    """Count the total number of rows in a CSV file without loading the entire file into memory.
    Args:
        file_path (str): The path to the CSV file.
        chunk_size (int): The number of rows to read at a time (default is 1000).
    Returns:
        int: The total number of rows in the CSV file.
    """
    total_rows = 0
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            total_rows += len(chunk)
        logger.info(f"Total rows counted successfully: {total_rows}")
        return total_rows
    except FileNotFoundError:
        logger.error(f"No file found at path: {file_path}")
        return 0
    except pd.errors.EmptyDataError:
        logger.error(f"The file at path: {file_path} is empty.")
        return 0
    except Exception as e:
        logger.error(f"An error occurred while counting rows in the CSV file: {e}")
        return 0
