# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:57:39 2020

@author: GSCA
"""

import pandas as pd
import time
import numpy as np
import os
from datetime import datetime
from datetime import datetime, timedelta
from functions import write_predictions,modifie_df_for_fb,make_sliced_request,scale,inverse_difference,difference,invert_scale
from influxdb import InfluxDBClient
from functions import make_sliced_request
from statsmodels.tsa.seasonal import seasonal_decompose
from ml_functions import decoupe_dataframe
from tensorflow.keras.models import Sequential,save_model,load_model
from tensorflow.keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from Predictor import Predictor 



class Existing_Predictor(Predictor):
    
    def __init__(self,df,host,measurement,look_back,metric,nb_features,nb_epochs,nb_batch,form,freq_period) :
        Predictor.__init__(self)
        self.df=df
        self.host=host
        self.measurement=measurement
        self.form=form
        self.look_back=look_back
        self.freq_period=freq_period
        trend_x, trend_y,seasonal_x,seasonal_y,residual_x,residual_y=self.prepare_data(df,look_back,freq_period)
        model_trend,model_seasonal,model_residual=self.load_models()
        model_trend=self.train_model(model_trend,trend_x,trend_y,nb_epochs,nb_batch,"trend")
        model_seasonal=self.train_model(model_seasonal,seasonal_x,seasonal_y,nb_epochs,nb_batch,"seasonal")
        model_residual=self.train_model(model_residual,residual_x,residual_y,nb_epochs,nb_batch,"residual")
        self.model_trend=model_trend     
        self.model_seasonal=model_seasonal
        self.model_residual=model_residual

    def load_models(self):
        file=""
        for element in self.form :
            value=element.split("=")
            file=file+'_'+value[1]
        file=file[1 :].replace(":","")
        model_trend=load_model("Modeles/"+file+"_"+self.measurement+"/"+"trend"+".h5")
        model_seasonal=load_model("Modeles/"+file+"_"+self.measurement+"/"+"seasonal"+".h5")
        model_residual=load_model("Modeles/"+file+"_"+self.measurement+"/"+"residual"+".h5")
        return model_trend,model_seasonal,model_residual
    

    
