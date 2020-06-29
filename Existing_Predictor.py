
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from Predictor import Predictor 



class Existing_Predictor(Predictor):

    def __init__(self,df,host,measurement,look_back,metric,nb_features,nb_epochs,nb_batch,form,freq_period,file) :
        Predictor.__init__(self)
        self.df=df
        self.host=host
        self.measurement=measurement
        self.form=form
        self.look_back=look_back
        self.freq_period=freq_period
        self.file=file
        model_trend,model_seasonal,model_residual=self.load_models()
        self.model_trend=model_trend
        self.model_seasonal=model_seasonal
        self.model_residual=model_residual

    def load_models(self):
        file=self.file
        print(file+"/"+"trend"+".h5")
        model_trend=load_model(file+"/"+"trend"+".h5")
        model_seasonal=load_model(file+"/"+"seasonal"+".h5")
        model_residual=load_model(file+"/"+"residual"+".h5")
        print(model_trend.summary())
        return model_trend,model_seasonal,model_residual

    def make_prediction(self,df,c):
        import streamlit  as st
        trend_x, trend_y, seasonal_x, seasonal_y, residual_x, residual_y = self.prepare_data(df, self.look_back, self.freq_period)
        pred=np.zeros((1,len(trend_x)))
        real_val=np.zeros((1,len(trend_x)))
        latest_iteration = st.empty()
        bar = st.progress(0)
        for i in range(0, len(trend_x)):
            latest_iteration.text(f'Iteration {int(i/len(trend_x)*100) }')
            bar.progress(int(i/len(trend_x)*100))
            dataset = np.reshape(trend_x[i,:], (1, 1, self.look_back))
            prediction_trend = self.model_trend.predict(dataset)
            dataset = np.reshape(residual_x[i,:], (1, 1, self.look_back))
            prediction_residual = self.model_residual.predict(dataset)
            dataset = np.reshape(seasonal_x[i,:], (1, 1, self.look_back))
            prediction_seasonal = self.model_seasonal.predict(dataset)
            prediction=prediction_trend+prediction_seasonal+prediction_residual
            real_value=trend_y[i]+seasonal_y[i]+residual_y[i]
            pred[0,i]=prediction
            real_val[0,i]=real_value
            if abs(prediction-real_value)[0,0] > c :
                plt.scatter(i,real_val[0,i],color="red",marker="x")
        plt.plot(pred[0,:],label="prediction")
        plt.plot(real_val[0,:],label="vrai valeur")
        plt.legend()
        plt.show()

    
