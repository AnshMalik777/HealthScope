
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "fin_clean.xlsx"


def load_healthcare_data():
    """Load the healthcare dataset."""
    df = pd.read_excel(DATA_PATH, engine="openpyxl")

    # Convert numeric fields safely
    numeric_columns = [
        "ages",
        "bmi",
        "days_hospitalized",
        "treatment_cost_inr",
        "recovery_days",
        "death_flag"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_healthcare_overview(df):
    """Calculate key healthcare statistics."""

    total_records = len(df)

    deaths = int(df["death_flag"].fillna(0).sum())

    mortality_rate = (
        (deaths / total_records) * 100
        if total_records else 0
    )

    average_age = df["ages"].mean()
    average_cost = df["treatment_cost_inr"].mean()
    average_recovery = df["recovery_days"].mean()

    disease_count = df["disease_name"].nunique()
    state_count = df["state"].nunique()

    most_common_disease = (
        df["disease_name"].mode().iloc[0]
        if not df["disease_name"].mode().empty
        else "Not available"
    )

    most_affected_state = (
        df["state"].mode().iloc[0]
        if not df["state"].mode().empty
        else "Not available"
    )

    return {
        "total_records": total_records,
        "total_deaths": deaths,
        "mortality_rate_percent": round(mortality_rate, 2),
        "average_age": round(average_age, 2),
        "average_treatment_cost_inr": round(average_cost, 2),
        "average_recovery_days": round(average_recovery, 2),
        "unique_diseases": disease_count,
        "unique_states": state_count,
        "most_common_disease": most_common_disease,
        "most_affected_state": most_affected_state
    }


if __name__ == "__main__":
    print("Loading HealthScope dataset...")

    data = load_healthcare_data()
    results = get_healthcare_overview(data)

    print("\n========== HEALTHCARE OVERVIEW ==========")

    for key, value in results.items():
        print(f"{key}: {value}")

    print("\nAnalysis completed successfully!")
