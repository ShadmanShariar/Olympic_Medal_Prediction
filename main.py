import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Reading the csv data file
teams = pd.read_csv("teams.csv")
print(teams)

# Taking importants col
teams = teams[["team","country","year","athletes","age","prev_medals","medals"]]
print(teams)

# Checking which col is best for co-relation with medals in lenear reg
print(teams.corr()["medals"])

# Graph
sns.lmplot(x='athletes',y='medals',data=teams,fit_reg=True, ci=None)
plt.show()
sns.lmplot(x='age',y='medals',data=teams,fit_reg=True, ci=None)
plt.show()

# Deleting Nulls
teams = teams.dropna()
print(teams)

#Filtering our data 1 for train out model another for test our model

train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()

# printing the size of train data and test data
print(train.shape)
print(test.shape)

#Importing our linear reg model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

# Train starts from here
predictors = ["athletes", "prev_medals"]

reg.fit(train[predictors], train["medals"])
LinearRegression()

#Printing the prediction
predictions = reg.predict(test[predictors])
print(predictions)


test["predictions"] = predictions
print(test)

test.loc[test["predictions"] < 0, "predictions"] = 0
print(test)

#test
print(test[test["team"] == "USA"])

#Printing the result
medals_by_team = test["medals"].groupby(test["team"]).mean()
print((medals_by_team))


