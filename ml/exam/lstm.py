import numpy as np
import matplotlib.pyplot as plt
import pandas
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler, MinMaxScaler

np.random.seed(5)


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def build_nn(x):
    model = Sequential()
    model.add(LSTM(120, input_shape=(x.shape[1], x.shape[2])))
    model.add(Dense(units=1))
    model.add(Activation('tanh'))
    model.summary()
    sgd = SGD(learning_rate=0.3)
    model.compile(optimizer=sgd, loss='mse')
    return model


def read_data(file_path):
    dataframe = pandas.read_csv(file_path, engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    return dataset


if __name__ == '__main__':

    train = read_data('training_data.csv')
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = StandardScaler()
    train = scaler.fit_transform(train)
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)

    test = read_data('final_result_data.csv')
    test = scaler.transform(test)
    testX, testY = create_dataset(test, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[1]))  # （样本个数，1，输入的维度）
    testX = np.reshape(testX, (testX.shape[0], look_back, testX.shape[1]))

    model = build_nn(trainX)
    # model.fit(trainX, trainY, epochs=100)
    # model.save('model_test.h5')

    from keras.models import load_model
    model = load_model('model_test.h5')

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    trainPredictPlot = np.empty_like(train)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    testPredictPlot = np.empty_like(train)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(train) - 1, :] = testPredict
    plt.plot(scaler.inverse_transform(train))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
