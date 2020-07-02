# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:07:22 2020

@author: GSCA
"""


import numpy as np
import os
import csv
from statsmodels.tsa.seasonal import seasonal_decompose
from Functions.ml_functions import decoupe_dataframe
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import streamlit as st
def transform_time(period):
    if period[-1] == "s":
        nb_points = int(period[0:len(period) - 1])
    elif period[-1] == "m":
        nb_points = int(60/int(period[0:len(period) - 1])) * 24 * 7
    elif period[-1] == "h":
        nb_points = int(period[0:len(period) - 1]) * 24 * 7
    elif period[-1] == "d":
        nb_points = int(period[0:len(period) - 1]) * 7
    elif period[-1] == "w":
        nb_points = int(period[0:len(period) - 1]) * 1
    return(nb_points)



class EarlyStoppingByUnderVal(Callback):
    '''
    Class to stop model's training earlier if the value we monitor(monitor) goes under a threshold (value)
    replace usual callbacks functions from keras 
    '''
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            #warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
            print("error")

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


class Simple_lstm_predictor():
    
    def __init__(self):
        print("Creating object")

    def prepare_data(self,df,look_back,freq_period):
        '''  
        Parameters
        ----------
        df : DataFrame
            datafrmae contening historical data .
        look_back : int
            length entry of the model .
        
        Decompose the signal in three sub signals, trend,seasonal and residual in order to work separetly on each signal

        Returns
        -------
        trend_x : array
             values of the trend of the signal, matrix of dimention X array of dimension (1,length entry of model) X= length(dataframe)/look_back.
        trend_y : array
            vaklues to be predicted during the training
        seasonal_x : array
            same as trend_x but with the seasonal part of the signal.
        seasonal_y : array
            same as trend_y but with the seasonal part of the signal.
        residual_x : array
            same as trend_x but with the residual part of the signal.
        residual_y : array
            same as trend_y but with the residual part of the signal.
        '''
        self.look_back=look_back
        df_a=df
        self.df=df
        print( self.freq_period)
        self.margin=self.frame_prediction(0,df)
        self.dic = str(self.margin) +','+ str(look_back)
        df=df.dropna()
        df=df.reset_index(drop=True)
        trend_x,trend_y=decoupe_dataframe(np.asarray(df["y"]), look_back)
        print("prepared")
        return trend_x, trend_y
    
    
    def train_model(self,model,x_train,y_train,nb_epochs,nb_batch,name):
        '''   
        Train the model and save it in the right file
        
        Parameters
        ----------
        model : Sequential object
            model.
        x_train : array
            training data inputs.
        y_train : array
            training data ouputs.
        nb_epochs : int.
            nb of training repetitions.
        nb_batch : int
            size of batch of element which gonna enter in the model before doing a back propagation.
        trend : bool
            Distinguish trend signal from others (more complicated to modelise).
        
        Returns
        -------
        model : Sequential object
            model.

        '''
        nb_epochs=nb_epochs*7
        x_train=x_train.reshape(x_train.shape[0],1,x_train.shape[1])
        hist=model.fit(x_train,y_train,epochs=nb_epochs,batch_size=nb_batch,verbose=2)
        i=0
        while hist.history["mse"][-1] > 10 and i <5:     #####################################§changer
            i=i+1
            epochs=1
            hist=model.fit(x_train,y_train,epochs=epochs,batch_size=100,verbose=2)
        print("model_trained")
        self.model_save(model,name)
        return model
    
    def model_save(self,model,name) :
       file=""
       print(self.form)
       if type(self.form) != list :
           form=self.form[1 :].split(",")
       else :
           form = self.form
       try :
           for element in form :
               value=element.split("=")
               file=file+'_'+value[1]
           file=file[1 :].replace(":","")
       except Exception :
           file=self.host
       file=file.replace(" ","")
       path="Modeles/"+file+"_"+self.measurement

       if not os.path.isdir(path) :
           os.makedirs(path)
       path="Modeles/"+file+"_"+self.measurement+"/"+name+".h5"
       save_model(model,path)
       with open("Modeles/"+file+"_"+self.measurement+"/"+'/dict.txt', 'w') as txt_file:
           txt_file.write(str(self.dic))

    def frame_prediction(self, prediction, df):
        '''
        This function compute the 95% confident interval by calculating the standard deviation and the mean of the residual(Gaussian like distribution)
        and return yestimated +- 1.96*CI +mean (1.96 to have 95%)

        Parameters
        ----------
        prediction : array
            array contening the perdicted vector (1,N) size.

        Returns
        -------
        lower : array
            array contening the lowers boundry values (1,N) size.
        upper : array
            array contening the upper boundry values (1,N) size.

        '''
        decomposition = seasonal_decompose(df["y"], period=self.freq_period)
        mae = -1 * np.mean(decomposition.resid)
        std_deviation = np.std(decomposition.resid)
        sc = 1.96  # 1.96 for a 95% accuracy
        margin_error = mae + sc * std_deviation
        lower = prediction - margin_error
        upper = prediction + margin_error
        return margin_error

    """ def make_prediction(self, c):
        print("Doing prediction")
        trend_x, trend_y = self.prepare_data(self.df, self.look_back, self.freq_period)
        pred = np.zeros((1, len(trend_x)))
        real_val = np.zeros((1, len(trend_x)))
        latest_iteration = st.empty()
        bar = st.progress(0)
        for i in range(0, len(trend_x)):
            latest_iteration.text(f'Iteration {int(i / len(trend_x) * 100)}')
            bar.progress(int(i / len(trend_x) * 100))
            dataset = np.reshape(trend_x[i, :], (1, 1, self.look_back))
            prediction_trend = self.model_trend.predict(dataset)
            prediction = prediction_trend
            lower, upper = self.frame_prediction(prediction, self.df["y"])
            real_value = self.df[i]
            pred[0, i] = prediction
            real_val[0, i] = real_value
            if abs(lower - real_value)[0, 0] > c or abs(upper - real_value)[0, 0] > c:
                plt.scatter(i, real_val[0, i], color="red", marker="x")
        plt.plot(pred[0, :], label="prediction")
        plt.fill_between(lower, upper, alpha=0.5)
        plt.plot(real_val[0, :], label="vrai valeur")
        plt.legend()
        plt.show()
        st.pyplot()
    
    
    def frame_prediction(self,prediction,df):
        '''
        This function compute the 95% confident interval by calculating the standard deviation and the mean of the residual(Gaussian like distribution)
        and return yestimated +- 1.96*CI +mean (1.96 to have 95%)

        Parameters
        ----------
        prediction : array
            array contening the perdicted vector (1,N) size.

        Returns
        -------
        lower : array
            array contening the lowers boundry values (1,N) size.
        upper : array
            array contening the upper boundry values (1,N) size.

        '''
        decomposition = seasonal_decompose(df["y"], period=self.freq_period)
        mae=-1*np.mean(decomposition.resid)
        std_deviation=np.std(decomposition.resid)
        sc = 1.96       #1.96 for a 95% accuracy
        margin_error = mae + sc * std_deviation
        lower = prediction - margin_error
        upper = prediction + margin_error
        return lower,upper"""
