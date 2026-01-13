import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import load_model

model = load_model('model_test.h5')
dataframe = pd.read_csv('final_result_data.csv', engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

# 数据规范化
mean = np.mean(dataset, axis=0)  # 均值
dataset -= mean  # 训练集
std = np.std(dataset, axis=0) # 标准差
dataset /= std

x = dataset[:, 3:]
y = dataset[:, 2]


y_pred = model.predict(x)
X_data = [xdata[4] for xdata in x]
plt.scatter(X_data, y)
# y_pred *= std
plt.plot(X_data, y_pred, 'r-', lw=3)
plt.show()

