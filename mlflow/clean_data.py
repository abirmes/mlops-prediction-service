
import pandas as pd
import numpy as np
import os

from sklearn.impute import KNNImputer

data = pd.read_csv(
    "/mlflow/dataset-diabete-68e2810ab0d7e949117525.csv"
).drop("Unnamed: 0", axis=1)

# Imputation
data_copy = data.copy()
for col in data.columns:
    if col != "Pregnancies":
        data_copy[col] = data_copy[col].replace(0, np.nan)
data = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(data_copy), columns=data_copy.columns)

# Log transform
for col in ['BloodPressure','Insulin','DiabetesPedigreeFunction']:
    data[col] = np.log1p(data[col])

# Outliers simplifiés (garder logique précédente)
def cap_outliers(df, col, factor_upper=2, factor_lower=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + factor_upper*IQR
    lower = Q1 - factor_lower*IQR
    df[col] = np.where(df[col]>upper, upper, df[col])
    df[col] = np.where(df[col]<lower, lower, df[col])
    return df

for col in ['Glucose','BMI','Age']:
    data = cap_outliers(data, col)

# Feature engineering
data['BMI_Age'] = data['BMI'] * data['Age']
data['Glucose_Insulin'] = data['Glucose'] * data['Insulin']

# **SAVE THE CLEANED DATA HERE**
folder = "/app/cleaned_data"
os.makedirs(folder, exist_ok=True)
data.to_csv(f"{folder}/data.csv", index=False)
print(f"Cleaned data saved to {folder}/data.csv")
