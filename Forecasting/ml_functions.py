# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:13:26 2020

@author: GSCA
"""



import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM



def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def build_model(train,n_timesteps,n_features,n_outputs):
    train_x, train_y = create_dataset(train,n_timesteps)
    print(train_x)
    verbose = 0
    epochs = 70
    batch_size = 16
    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model

def decoupe_dataframe(df,look_back):
    dataX,dataY = [],[]
    for i in range(len(df) - look_back - 1):
        a = df[i:(i + look_back)]
        dataY=dataY+[df[i+look_back]]
        dataX.append(a)
    return (np.asarray(dataX),np.asarray(dataY).flatten())


