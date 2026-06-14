import pandas as pd
import logging

logger = logging.getLogger(__name__)


def calculated_total_sales(file_path: str) -> float:
    """Calculate total Revenue from a CSV file and log the process.
    Args:
        file_path (str): The path to the CSV file containing sales data.
    Returns:
        float: The total sales amount.
    Example:
        total_sales = calculated_total_sales("sales_data.csv")
        logger.info(f"Total sales: {total_sales}")
    """
    try:
        # 1. Loading data blindly
        df = pd.read_csv(file_path)
        logger.info("Data loaded successfully now calculating total sales.")
        # 2. Filtering with bad variable names
        df_filtered = df[df["Status"] == "Completed"]

        # 3. Doing math and printing
        total_sales = df_filtered["Amount"].sum()
        logger.info(f"Total sales: {total_sales}")
        return total_sales
    except FileNotFoundError:
        logger.error(f"No file  found at path: {file_path}")
        return 0.0
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return 0.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    total_sales = calculated_total_sales("sales_data.csv")
