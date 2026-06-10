import sys

sys.path.append(".")

import logging
import pandas as pd
from src.data_loader import load_csv, filter_by_value, get_missing_counts

logging.basicConfig(
    filename="test_all.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

print("--- Running Test 1: Invalid File Path ---")
result_file = load_csv("file_that_does_not_exist.csv")
print(f"Returned Value: {result_file}")

print("\n--- Running Test 2: Invalid Column Name ---")
df = pd.DataFrame({"A": [1, 2, 3]})
result_col = filter_by_value(df, "nonexistent_column", 5)
print(f"Returned Value: {result_col}")

print("\n--- Running Test 3: Invalid Column Name ---")
result_missing = get_missing_counts(pd.DataFrame())
print(f"Returned test 3 result:{result_missing}")
