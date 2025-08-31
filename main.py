import pandas as pd
from sklearn.ensemble import RandomForce

# getting file
matches = pd.read_csv('season-2425.csv')

# convert date collum into a date object
matches["Date"] = pd.to_datetime(matches["Date"])

# Adding a day of the week
matches["Day_Code"] = matches["Date"].dt.dayofweek

# Results
matches["Target"] = (matches["FTR"] == "H").astype('int')