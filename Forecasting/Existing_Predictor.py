# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:57:39 2020

@author: GSCA
"""


from tensorflow.keras.models import Sequential,save_model,load_model

from Predictor import Predictor
import pickle



class Existing_Predictor(Predictor):
    
    def __init__(self,df,host,measurement,look_back,metric,nb_features,nb_epochs,nb_batch,form,freq_period) :
        Predictor.__init__(self)
        self.df=df
        self.host=host
        self.measurement=measurement
        self.form=form
        self.look_back=look_back
        self.freq_period=freq_period
        trend_x, trend_y,seasonal_x,seasonal_y,residual_x,residual_y=self.prepare_data(df,look_back,freq_period,self.form)
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
        model_trend=load_model(r"../Modeles/"+file+"_"+self.measurement+"/"+"trend"+".h5")
        model_seasonal=load_model(r"../Modeles/"+file+"_"+self.measurement+"/"+"seasonal"+".h5")
        model_residual=load_model(r"../Modeles/"+file+"_"+self.measurement+"/"+"residual"+".h5")
        print("loaded")
        return model_trend,model_seasonal,model_residual
    

    
