import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# 简单线性回归
if __name__ == '__main__':
    # 读取数据
    dataset = pd.read_csv("final_training_data.csv")
    x = dataset.iloc[:, 3:].values
    y = dataset.iloc[:, 2].values

    onehotencoder = OneHotEncoder()
    x = onehotencoder.fit_transform(x).toarray()
    x = x[:, 1:]
    # 划分数据集
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=1/4, random_state=0)
    # 训练
    lr = LinearRegression()
    regressor = lr.fit(X_train, Y_train)
    # 预测
    Y_pred = regressor.predict(X_test)
    print()
    # print(regressor.predict([[990, 1]]))
