

from app.validation.validate import validate_Data
import pandas as pd

data = pd.read_csv('app/cleaned_data/data.csv')
total = validate_Data(data)
print(total)