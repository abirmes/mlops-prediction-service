import pandas as pd
import numpy as np

data = pd.read_csv("./mlflow/data/cleaned_data/data.csv")


def validate_value(count: int) -> int:
    return 1 if count > 0 else 0


def validate_data_types(df: pd.DataFrame) -> int:
    EXPECTED_SCHEMA = {
        "Pregnancies": np.floating,
        "Glucose": np.floating,
        "BloodPressure": np.floating,
        "SkinThickness": np.floating,
        "Insulin": np.floating,
        "BMI": np.floating,
        "DiabetesPedigreeFunction": np.floating,
        "Age": np.floating,
    }

    for col, expected_type in EXPECTED_SCHEMA.items():
        if col not in df.columns:
            return 1
        if not np.issubdtype(df[col].dtype, expected_type):
            return 1
    return 0


def age_validation(df: pd.DataFrame) -> int:
    if not df["Age"].between(18, 75).all():
        return 1
    return 0


def validate_data(df: pd.DataFrame) -> int:
    total = 0

    # Null values
    total += validate_value(df.isna().sum().sum())

    # Duplicates
    total += validate_value(df.duplicated().sum())

    # Schema
    total += validate_data_types(df)

    # Business rule
    total += age_validation(df)

    return total


if __name__ == "__main__":
    print("Data quality score:", validate_data(data))
