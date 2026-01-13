import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 第三天，多元线性回归
if __name__ == '__main__':
    dataset = pd.read_csv("../datasets/ 50_Startups.csv")
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, 4].values

    onehotencoder = OneHotEncoder()
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:, 1:]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)
    y_pred = regressor.predict(X_test)

