import pandas as pd
import logging

logger = logging.getLogger(__name__)


# using f-string formatting:
# === Dataset Quality Report ===
# Shape: 1,000 rows × 8 cols
# Missing: 45 values (0.56%)
# Duplicates: 3 rows
# Numeric cols: age, salary, years_exp
# # ==============================
def generate_quality_report(df: pd.DataFrame) -> str:
    """Generates a quality report for the given DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to analyze.
    Returns:
        str: A formatted quality report.
    """
    rows, cols = df.shape
    total_values = rows * cols
    missing_values = df.isnull().sum().sum()
    missing_percentage = (missing_values / total_values) * 100
    duplicate_rows = df.duplicated().sum()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    report = (
        f"=== Dataset Quality Report ===\n"
        f"Shape:        {rows:,.0f} rows × {cols:,.0f} cols\n"
        f"Missing:      {missing_values:>9,.0f} values ({missing_percentage:.2%})\n"
        f"Duplicates:   {duplicate_rows:>9,.0f} rows\n"
        f"Numeric cols: {",".join(numeric_cols)}\n"
        f"=============================="
    )
    return report


def export_to_text(report: str, filepath: str) -> None:
    """Exports the given report string to a text file.
    Args:
        report (str): The report content to write to the file.
        filepath (str): The path where the text file will be saved.
    Example:
        report = generate_quality_report(df)
        export_to_text(report, "quality_report.txt")
    """
    try:
        with open(filepath, "w") as file:
            file.write(report)
            logger.info(f"Report Successfully exported to {filepath}")
    # add specific exception handling for file-related errors
    except FileNotFoundError:
        logger.error(f"Error: The file path {filepath} was not found.")
    except IOError as e:
        logger.error(f"IOError while writing to file {filepath}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while exporting the report: {e}")


if __name__ == "__main__":
    # Example usage:
    df = pd.DataFrame(
        {
            "age": [25, 30, 22, None, 28],
            "salary": [50000, 60000, 55000, 52000, None],
            "years_exp": [2, 5, 3, 4, None],
        }
    )
    report = generate_quality_report(df)
    print(report)
    export_to_text(report, "quality_report.txt")
