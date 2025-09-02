import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

# getting file
matches = pd.read_csv('season-2425.csv')

# convert date collum into a date object
matches["Date"] = pd.to_datetime(matches["Date"], format="%d/%m/%y", dayfirst=True)

# Adding a day of the week
matches["Day_Code"] = matches["Date"].dt.dayofweek

# Which opponenet
matches["Opponent_Code"] = matches["AwayTeam"].astype("category").cat.codes

# Creating Groups
grouped_matches = matches.groupby("HomeTeam")

# Results
matches["Target"] = (matches["FTR"] == "H").astype('int')

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = matches[matches["Date"] < "2025-01-01"]
test = matches[matches["Date"] > "2025-01-01"]
predictors = ["Day_Code", "Opponent_Code"]

rf.fit(train[predictors], train["Target"])
RandomForestClassifier(min_samples_split=10, n_estimators=50, random_state=1)
preds = rf.predict(test[predictors])


# Testing out the accuracy
acc = accuracy_score(test["Target"], preds)
print(acc)

# Dataframe
combined = pd.DataFrame(dict(actual=test["Target"], prediction=preds))
print(pd.crosstab(index=combined["actual"], columns=combined["prediction"]))


# group = grouped_matches.get_group("Barcelona") (test)

# Rolling averages
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("Date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

# Getting data to test out and improve accuracy of model
cols = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
new_cols = [f"{c}_rolling" for c in cols]

# Creating rolling dataset
matches_rolling = matches.groupby("HomeTeam", group_keys=False).apply(lambda x: rolling_averages(x, cols, new_cols))

# Clearing any duplicates
matches_rolling.index = range(matches_rolling.shape[0])

print(matches_rolling)
