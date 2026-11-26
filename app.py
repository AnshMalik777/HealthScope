
import pandas as pd
from pathlib import Path

# Locate the healthcare Excel dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "fin_clean.xlsx"
# Load the dataset
try:
    df = pd.read_excel(DATA_PATH, engine="openpyxl")
    print("\n================================")
    print("      HEALTHSCOPE DATA CHECK")
    print("================================")

    print(f"\nTotal Records: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")

    print("\nColumn Names:")
    for column in df.columns:
        print("-", column)

    print("\nFirst 5 Records:")
    print(df.head().to_string())

    print("\nMissing Values:")
    print(df.isnull().sum().to_string())

    print("\nData loaded successfully!")

except FileNotFoundError:
    print("ERROR: Excel file not found.")
    print("Check that fin.xlsx is inside the data folder.")

except Exception as e:
    print("ERROR:", e)
