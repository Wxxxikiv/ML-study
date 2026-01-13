import numpy as np
import matplotlib.pyplot as plt
import pandas
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler

np.random.seed(5)


def build_nn():
    model = Sequential()
    model.add(Dense(units=5, input_dim=5))
    # model.add(layers.BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dense(units=1024))
    model.add(Activation('tanh'))
    # model.add(Dropout(0.5))
    model.add(Dense(units=512))
    model.add(Activation('tanh'))
    # model.add(Dropout(0.5))
    model.add(Dense(units=32))
    model.add(Activation('tanh'))
    model.add(Dense(units=1))
    model.add(Activation('tanh'))
    model.summary()
    sgd = SGD(learning_rate=0.3)
    model.compile(optimizer=sgd, loss='mse')
    return model


def read_data(file_path):
    dataframe = pandas.read_csv(file_path, engine='python', usecols=[2, 3, 4, 5, 6, 7])
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    return dataset[:, 1:], dataset[:, 0:1]


if __name__ == '__main__':

    x, y = read_data('final_training_data.csv')

    x_stand_scaler = StandardScaler()
    y_stand_scaler = StandardScaler()
    x = x_stand_scaler.fit_transform(x)
    y = y_stand_scaler.fit_transform(y)

    model = build_nn()
    model.fit(x, y, epochs=100)
    model.save('model_test.h5')

    from keras.models import load_model
    model = load_model('model_test.h5')

    x, y = read_data('final_result_data.csv')

    x_plt = [t[4] for t in x]
    x = x_stand_scaler.transform(x)

    y_pred = model.predict(x)
    y_pred = y_stand_scaler.inverse_transform(y_pred)

    plt.scatter(x_plt, y)
    plt.plot(x_plt, y_pred, 'r-', lw=3)
    plt.show()
