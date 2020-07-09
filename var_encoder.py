# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:07:22 2020

@author: GSCA
"""


import numpy as np
from Functions.ml_functions import decoupe_dataframe
from tensorflow.keras.models import save_model,Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout,TimeDistributed,RepeatVector
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
from sklearn.externals import joblib

class VAR_LSTM():
    def __init__(self,look_back,form,measurement,host):
        print("Creating variable encoder object ")
        self.look_back=look_back
        self.form=form
        self.measurement=measurement
        self.host=host




    def make_prediction(self,x_real,model,threshold,scaler):
        Y = scaler.transform(np.reshape(np.array(x_real), (-1, 1)))
        x_real,_=decoupe_dataframe(Y,168)
        x_real = x_real.reshape(x_real.shape[0], 1, x_real.shape[1])
        prediction=model.predict(x_real)
        test_mae_loss = np.mean(np.abs(prediction - x_real), axis=1)
        anomalies=[test_mae_loss[i] > threshold/10 for i in range(len(test_mae_loss))]
        return anomalies

    def model_save(self, model,scaler,name="var"):
        file = ""
        if type(self.form) != list:
            form = self.form[1:].split(",")
        else:
            form = self.form
        try:
            for element in form:
                value = element.split("=")
                file = file + '_' + value[1]
            file = file[1:].replace(":", "")
        except Exception:
            file = self.host
        file = file.replace(" ", "")
        path = "Modeles_AD/" + file + "_" + self.measurement
        self.dic =str(self.look_back)
        if not os.path.isdir(path):
            os.makedirs(path)
        path = "Modeles_AD/" + file + "_" + self.measurement + "/" + name + ".h5"
        save_model(model, path)
        with open("Modeles_AD/" + file + "_" + self.measurement + "/" + 'params_var_model.txt', 'w') as txt_file:
            txt_file.write(str(self.dic))
        scalerfile = "Modeles_AD/" + file + "_" + self.measurement + "/" +'scaler.pkl'
        joblib.dump(scaler, scalerfile)

class New_VAR_LSTM(VAR_LSTM):
    def __init__(self,look_back,nb_pas,df,form,measurement,host):
        print("New VAR_LSTM is being created")
        VAR_LSTM.__init__(self,look_back,form,measurement,host)
        self.model=self.make_model(look_back)
        self.nb_pas=nb_pas
        self.form=form
        self.measurement=measurement
        self.host=host
        scaler_fitted,self.history=self.train_model(df,self.look_back,self.model)
        self.model_save(self.model,scaler_fitted)

    def make_model(slef, look_back):
        model = Sequential()
        model.add(LSTM(units=128, input_shape=(1, look_back), return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=64, input_shape=(1, look_back)))
        model.add(RepeatVector(n=1))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=128, input_shape=(1, look_back), return_sequences=True))
        model.add(TimeDistributed(Dense(units=1)))
        model.compile(loss='mse', optimizer='adam')
        return model

    def train_model(self, df, look_back, model):
        scaler = MinMaxScaler()
        Y = scaler.fit_transform(np.reshape(np.array(df["y"]),(-1,1)))
        x_train, y_train = decoupe_dataframe(np.reshape(Y,(-1,1)), look_back)
        print(np.shape(x_train))
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
        history = model.fit(
            x_train, y_train,
            epochs=500,
            batch_size=30,
            validation_split=0.1,
            shuffle=False
        )
        return scaler, history

class Existing_VAR_LSTM(VAR_LSTM):
    def __init__(self,look_back,file,form,measurement,host):
        VAR_LSTM.__init__(self,look_back,form,measurement,host)
        self.file=file


    def load_models(self):
        file=self.file
        model_trend=load_model(file)
        with open(self.file.split("/")[0]+"/"+self.file.split("/")[1]+'/params_var_model.txt', 'r') as txt_file:
            dic=txt_file.read()
        self.look_back=int(dic[0])
        scalerfile = self.file.split("/")[0] + "/" + self.file.split("/")[1] + "/" + 'scaler.pkl'
        scaler = joblib.load(scalerfile)
        return model_trend,scaler
