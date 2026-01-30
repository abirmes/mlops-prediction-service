import pandas as pd 

data = pd.read_csv("" \
"app/mlflow/cleaned_data.csv")
null = data.isna().sum()
def validate_Data() :
    total = 0 
    total += validate_Data( data.isna().sum())
    duplates += validate_Data(data.duplicated().sum())
    total += validate_Data(duplates)
    total += validate_Data_types(data)
    total += age_validation(data)

    return total

def validate_Value (count : int) :
    if count > 0 :
        return 1 
def  validate_Data_types(data) :
    EXPECTED_SCHEMA = {
    "Pregnancies": float,
    "Glucose": float,
    "BloodPressure": float,
    "SkinThickness": float,
    "Insulin": float,
    "BMI": float,
    "DiabetesPedigreeFunction": float,
    "Age": float,
}
    for col in data.columns : 
        if col in EXPECTED_SCHEMA :
            if   col.dtype != EXPECTED_SCHEMA[col] :
                return 1 
    else : return 0 
def age_validation(data) :
    if not data['age'].between(18, 75).all():
        return 1 
    return 0 



