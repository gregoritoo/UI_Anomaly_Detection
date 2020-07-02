
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from Simple_lstm_predictor import Simple_lstm_predictor
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
import csv

class Simple_lstm_existing_predictor(Simple_lstm_predictor):

    def __init__(self,df,host,measurement,c,metric,nb_features,nb_epochs,nb_batch,form,freq_period,file) :
        Predictor.__init__(self)
        self.df=df
        self.host=host
        self.measurement=measurement
        self.form=form
        self.c=c
        self.freq_period=1
        self.file=file
        model_trend=self.load_models()
        self.model_trend=model_trend


    def load_models(self):
        file=self.file
        print(file+"/"+"trend"+".h5")
        model_trend=load_model("r"+file+"/"+"trend"+".h5")
        print(model_trend.summary())
        with open("r"+self.file+'/dict.txt', 'r') as txt_file:
            dic=txt_file.read()
        dic=dic.split(',')
        print(dic)
        self.margin_error=float(dic[0])
        self.look_back=int(dic[1])
        return model_trend


    def make_prediction(self,df,c):
        import streamlit  as st
        trend_x, trend_y = self.prepare_data(df, self.look_back, self.freq_period)
        pred=np.zeros((1,len(trend_x)))
        real_val=np.zeros((1,len(trend_x)))
        latest_iteration = st.empty()
        bar = st.progress(0)

        for i in range(0, len(trend_x)):
            latest_iteration.text(f'Iteration {int(i/len(trend_x)*100) }')
            bar.progress(int(i/len(trend_x)*100))
            dataset = np.reshape(trend_x[i,:], (1, 1, self.look_back))
            prediction_trend = self.model_trend.predict(dataset)
            pred[0,i]=prediction_trend
            real_val[0,i]=df["y"][i-1]
            if abs(prediction_trend - df["y"][i]) > float(self.margin_error) + self.c :
                plt.scatter(i+1,real_val[0,i],color="black",marker="x")
        plt.plot(pred[0,:],label="prediction")
        plt.plot(pred[0,:]+float(self.margin_error),label="upper")
        plt.plot(pred[0,:]-float(self.margin_error),label="lower")
        plt.plot(real_val[0,:],label="vrai valeur")
        plt.legend()
        plt.show()
