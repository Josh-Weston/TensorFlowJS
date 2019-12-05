import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow import keras
Dense = keras.layers.Dense
LSTM = keras.layers.LSTM
Sequential = keras.models.Sequential
Dropout = keras.layers.Dropout

##from keras.layers import Dense
##from keras.layers import LSTM
##from keras.models import Sequential

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

n_epochs = 10

# load the dataset
data = pd.read_csv('./data/international-airline-passengers.csv',
                   usecols=[1],
                   skiprows=[145,146])

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# shift Y forward
look_back = 1
dataX = np.array(data).flatten()[:-look_back]
dataY = np.array(data).flatten()[look_back:]

# split into train and test sets
train_size = int(len(data) * 2/3)
trainX = dataX[:train_size]
trainY = dataY[:train_size]
testX = dataX[train_size - 1:]
testY = dataY[train_size - 1:]

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], look_back, 1))
testX = np.reshape(testX, (testX.shape[0], look_back, 1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(32, input_shape=(1, look_back)))
model.add(Dropout(.1))
model.add(Dense(32))
model.add(Dropout(.1))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=n_epochs, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions (unscale back)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: {} RMSE'.format(trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: {} RMSE'.format(testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[:len(trainPredict)] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) - 1:len(trainPredict) + len(testPredict) - 1] = testPredict

# plot baseline and predictions
fig = plt.figure()
plt.plot(scaler.inverse_transform(data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
fig.savefig('lstm.png')
plt.show()

print(testX)
