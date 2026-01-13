import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 第二天，简单线性回归
if __name__ == '__main__':
    # 读取数据
    dataset = pd.read_csv("../datasets/studentscores.csv")
    x = dataset.iloc[:, :1].values
    y = dataset.iloc[:, 1].values
    # 划分数据集
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=1/4, random_state=0)
    # 训练
    lr = LinearRegression()
    regressor = lr.fit(X_train, Y_train)
    # 预测
    Y_pred = regressor.predict(X_test)
    # 训练集结果可视化
    plt.scatter(X_train, Y_train, color='red')
    plt.plot(X_test, regressor.predict(X_test), color='blue')
    plt.show()

    # 测试集结果可视化
    # plt.scatter(X_test, Y_test, color='red')
    # plt.plot(X_test, regressor.predict(X_test), color='blue')
    # plt.show()
