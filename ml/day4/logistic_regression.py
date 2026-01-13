import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 逻辑回归
if __name__ == '__main__':
    dataset = pd.read_csv("../datasets/Social_Network_Ads.csv")
    X = dataset.iloc[:, [2, 3]].values
    Y = dataset.iloc[:, 4].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=0)

    # 特征缩放，加快模型收敛速度
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.linear_model import LogisticRegression
    classifier  = LogisticRegression()
    classifier .fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # 生成混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    # 可视化
    from matplotlib.colors import ListedColormap
    # X_set, y_set = X_train, y_train
    # X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    #                      np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    # plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    #              alpha=0.75, cmap=ListedColormap(('red', 'green')))
    # plt.xlim(X1.min(), X1.max())
    # plt.ylim(X2.min(), X2.max())
    # for i, j in enumerate(np.unique(y_set)):
    #     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    #                 c=ListedColormap(('red', 'green'))(i), label=j)
    #
    # plt.title(' LOGISTIC(Training set)')
    # plt.xlabel(' Age')
    # plt.ylabel(' Estimated Salary')
    # plt.legend()
    # plt.show()

    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)

    plt.title(' LOGISTIC(Test set)')
    plt.xlabel(' Age')
    plt.ylabel(' Estimated Salary')
    plt.legend()
    plt.show()

