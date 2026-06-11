import pandas as pd


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
