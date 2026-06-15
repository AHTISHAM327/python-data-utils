import pandas as pd
import logging

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
            f"Succesfully dropped null values. Original shape:{df.shape},New Shape:{clean_df.shape}"
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
            f"Successfully removed the duplicates.Original shape:{df.shape}, new shape : {clean_df.shape}"
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


if __name__ == "__main__":
    logging.basicConfig(
        filename="data_cleaner.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.info("Data cleaner module executed directly.")
