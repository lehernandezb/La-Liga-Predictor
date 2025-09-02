import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# getting file
matches = pd.read_csv('season-2425.csv')

# convert date collum into a date object
matches["Date"] = pd.to_datetime(matches["Date"], format="%d/%m/%y", dayfirst=True)

# Adding a day of the week
matches["Day_Code"] = matches["Date"].dt.dayofweek

# Which opponenet
matches["Opponent_Code"] = matches["AwayTeam"].astype("category").cat.codes

# Results
matches["Target"] = (matches["FTR"] == "H").astype('int')

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = matches[matches["Date"] < "2025-01-01"]
test = matches[matches["Date"] > "2025-01-01"]
predictors = ["Day_Code", "Opponent_Code"]

rf.fit(train[predictors], train["Target"])
RandomForestClassifier(min_samples_split=10, n_estimators=50, random_state=1)
preds = rf.predict(test[predictors])


# Testing out the accurcey
acc = accuracy_score(test["Target"], preds)
print(acc)
