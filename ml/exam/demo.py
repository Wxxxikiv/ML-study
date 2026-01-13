import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(5)

# load the dataset
dataframe = pandas.read_csv('final_training_data.csv', engine='python', usecols=[2, 3, 4, 5, 6, 7])
dataset = dataframe.values
dataset = dataset.astype('float32')

x_stand_scaler = StandardScaler()
y_stand_scaler = StandardScaler()
# dataset = stand_scaler.fit_transform(dataset) # 标准化

x = dataset[:, 1:]
x = x_stand_scaler.fit_transform(x)
y = dataset[:, 0:1]
y = y_stand_scaler.fit_transform(y)

# create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(units=5, input_dim=5))
model.add(layers.BatchNormalization())
model.add(Activation('tanh'))
model.add(Dense(units=1024))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(units=512))
model.add(Activation('tanh'))
model.add(Dense(units=32))
model.add(Activation('tanh'))
model.add(Dense(units=1))
model.add(Activation('tanh'))
model.summary()
sgd = SGD(learning_rate=0.3)
model.compile(optimizer=sgd, loss='mse')

# model.fit(x, y, epochs=50)
# model.save('model_test.h5')

from keras.models import load_model
model = load_model('model_test.h5')
dataframe = pandas.read_csv('final_result_data.csv', engine='python', usecols=[2, 3, 4, 5, 6, 7])
dataset = dataframe.values
dataset = dataset.astype('float32')

x_plt = dataset[:, 5]
x = dataset[:, 1:]
x = x_stand_scaler.transform(x)
y = dataset[:, 0:1]


y_pred = model.predict(x)
y_pred = y_stand_scaler.inverse_transform(y_pred)

plt.scatter(x_plt, y)
plt.plot(x_plt, y_pred, 'r-', lw=3)
plt.show()
