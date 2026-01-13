import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 导入数据集
def load_data():
    dataset = pd.read_csv("../datasets/Data.csv")
    x_data = dataset.iloc[:, :-1].values
    y_data = dataset.iloc[:, 3].values
    return x_data, y_data


# 处理丢失数据
def handling_lost_data(x):
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    x[:, 1:3] = imputer.fit_transform(x[:, 1:3])


# 解析分类数据
def parsing_categorical_data(x, y):
    labelencoder_X = LabelEncoder()
    x[:, 0] = labelencoder_X.fit_transform(x[:, 0])
    onehotencoder = OneHotEncoder()
    x = onehotencoder.fit_transform(x).toarray()
    labelencoder_Y = LabelEncoder()
    y = labelencoder_Y.fit_transform(y)


if __name__ == '__main__':
    x, y = load_data()
    handling_lost_data(x)
    # 数据集拆分
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # 特征量化
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
