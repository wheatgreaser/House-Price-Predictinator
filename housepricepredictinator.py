import pandas
from sklearn import linear_model

df = pandas.read_csv("housepricedata.csv")

Y = df[["SalePrice"]]
Y = Y.head(100)

X = df[["LotArea", "OverallCond", "YearBuilt"]]
X = X.head(100)

reg = linear_model.LinearRegression()
reg.fit(X, Y)

print(reg.predict([[8450, 7, 2002]]))





