

import pandas as pd
import time
import numpy as np
import pickle
from datetime import datetime
from Alert import Alert
from datetime import datetime, timedelta
from functions import write_predictions,modifie_df_for_fb,make_sliced_request,evaluate_linearity,train_linear_model,make_sliced_request,modifie_df_for_fb,make_form,transform_time,make_sliced_request_multicondition
from influxdb import InfluxDBClient
from statsmodels.tsa.seasonal import seasonal_decompose
from New_Predictor import New_Predictor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from  scipy import stats
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from numpy import diff
import sys

def predict(df,form,len_prediction,host,measurement,db,severity,freq_period,look_back,force_ml_model):
    '''
    This function receive the data ,evaluate how well a linear model can model the data.
    If it can then it modelise it by a linear function and if not then it trains a ML model.
    Then returns predictions

    Parameters
    ----------
    df : dataframe
        data.
    form : string
        string contening all the condition in a dict form ex (host : SUP-DEV-GSCA, device : sda) which is necessary to determine the model.
    len_prediction : int
        lenght we want to predict.
    host : str
        host name.
    measurement : str
        measurement to predict (mem,cpu,disk).
    db : str
        database's name .

    Returns
    -------
    prediction : array
        array contenaining the predicted values .
    lower : array
        array containing the lower incertitude interval's values .
    upper : array
        array containing the upper incertitude interval's values.

    '''
    r_sq=evaluate_linearity(df)
    print(r_sq)
    if force_ml_model == "Yes" :
        r_sq=0
    else :
        r_sq=1
    if r_sq > 0.80 :
        model=train_linear_model(df,86,severity)
        prediction=model.predict(np.linspace(len(df),len(df)+len_prediction-1,len_prediction).reshape(-1, 1)).reshape(1,-1)
        upper=prediction
        lower=prediction
    else :
        Future=New_Predictor(df=df,
                         host=host,
                         measurement=measurement,
                         look_back=look_back,
                         nb_layers=50 ,  
                         loss="mape",
                         metric="mse",
                         nb_features=1,
                         optimizer="Adamax",
                         nb_epochs=300,
                         nb_batch=100,
                         form=form,
                         freq_period=freq_period)
        prediction,lower,upper=Future.make_prediction(len_prediction)
    return prediction,lower,upper

class Time():
    def __init__(self,time):
        self.influx_time=time 
        if time[-1]=="m" :
            self.pandas_time=time[: -1]+"min"
        else :
            self.pandas_time=time
    

import time
list_params=sys.argv[1]
severity=2
list_params=list_params.split("' '")
form=list_params[0][1 :]
parameters=list_params[1].split("/")
period=parameters[0]
host=parameters[1]
measurement=parameters[2]
db=parameters[3]
freq_period=int(parameters[4])
gb=parameters[5]
field=list_params[2]
typo=list_params[3].replace(" ","")
cond=list_params[4].replace("/","'")
force_ml_model=list_params[5]
time_obj=Time(period)

#doit faire une semaine en générale suivant la périodicité supposé du signal(à calculer en fonvtion de la féquence d'échantillonage)
 #########a supprimer avant de mettre en prod

client_2 = InfluxDBClient(host="localhost", port=8086)
client_2.switch_database(db)
########## fin partie à supprimer 
dic=""
len_prediction=transform_time(period)*2
nb_week_to_query=13
look_back = transform_time(period)*1
df, client = make_sliced_request_multicondition(host, db, measurement, period, gb, cond, nb_week_to_query, typo,dic,field)
df = modifie_df_for_fb(df, typo)
df_a=df
plt.plot(df["y"])
plt.show(block=True)
prediction,lower,upper=predict(df,form,len_prediction,host,measurement,db,severity,freq_period,look_back,force_ml_model)
dates=pd.date_range(df["ds"][len(df)-12],freq=time_obj.pandas_time, periods=len_prediction)
df2=pd.DataFrame({"ds" : dates , "yhat" : np.transpose(prediction[0]),"yhat_upper" :np.transpose(upper[0]),"yhat_lower" :np.transpose(lower[0]) })
df=df.append(df2[ :]).reset_index()
write_predictions(df,client_2,measurement,host,db,form)
#########a supprimer avant de mettre en prod
cp = df_a[['ds', 'y']].copy()
cp["measurement"]="Historical_"+measurement
client_2.delete_series(database=db,measurement="Historical_"+measurement,tags={ "tag" : form[1 :]})
lines = [str(cp["measurement"][d]) 
                         + ",type=forecast"
                         + form
                         + " " 
                         + "real_y=" + str(cp["y"][d]) + " " + str(int(time.mktime(datetime.strptime(str(cp['ds'][d]), "%Y-%m-%d %H:%M:%S").timetuple()))) + "000000000" for d in range(len(cp))]
try :
    client_2.write(lines,{'db':db},204,'line')
except Exception :
    print("Les données n'ont pas été envoyées")
######### fin partie à supprimer 
alert=Alert(host,measurement)
alert.create('Attention le modèle associé à '+form.replace(","," AND ")+' dévie par rapport à la mémoire de : {{ index .Fields "performance_error" }}',form,period,typo,field)
alert.save_alert()
alert.launch()  

                    

           
