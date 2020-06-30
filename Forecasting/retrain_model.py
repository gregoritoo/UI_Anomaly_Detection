
import sys 
import os
import pandas as pd
import time
import numpy as np
import pickle
from datetime import datetime
from influxdb import InfluxDBClient
from datetime import timedelta 
from functions import make_sliced_request,write_predictions,modifie_df_for_fb,make_sliced_request_multicondition,modifie_df_for_fb,train_linear_model,transform_time
from Query_3 import Query_all
from Existing_Predictor import Existing_Predictor
from statsmodels.tsa.seasonal import seasonal_decompose
from Alert import Alert

def predict(df,form,len_prediction,host,measurement,db,severity,freq_period):
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
    if r_sq > 0.70 :
        model=train_linear_model(df,86,severity)
        prediction=model.predict(np.linspace(len(df),len(df)+len_prediction-1,len_prediction).reshape(-1, 1)).reshape(1,-1)
        upper=prediction
        lower=prediction
    else :
        Future=New_Predictor(df=df,
                         host=host,
                         measurement=measurement,
                         look_back=len_prediction,
                         nb_layers=50 ,  
                         loss="mape",
                         metric="mse",
                         nb_features=1,
                         optimizer="Adamax",
                         nb_epochs=300,
                         nb_batch=100,
                         form=form,
                         freq_period=7)
        prediction,lower,upper=Future.make_prediction(len_prediction)
    return prediction,lower,upper


class Time():
    def __init__(self,time):
        self.influx_time=time 
        if time[-1]=="m" :
            self.pandas_time=time[: -1]+"min"
        else :
            self.pandas_time=time


host=sys.argv[1]
measurement = sys.argv[2]
form = sys.argv[3]

period=sys.argv[4]
typo=sys.argv[5]
field=sys.argv[6]
time_obj=Time(period)
db="telegraf"
gb=""

look_back=transform_time(period)
len_prediction=look_back*12
dic=""
#########a supprimer avant de mettre en prod

client_2 = InfluxDBClient(host="localhost", port=8086)
client_2.switch_database(db)
cond=""


########## fin partie à supprimer 

len_historical_data=3
forma=form.split(",")

for element in form.split(",") :
    value=element.split("=")
    cond=cond+' AND "'+value[0]+'"=\''+value[1]+"'" 
    gb=gb+value[0]+","


df,client=make_sliced_request_multicondition(host,db,measurement,period,gb,cond,len_historical_data,typo,dic,field)
if type(df) == list :
    df=df[0]
df=modifie_df_for_fb(df,typo)
df_a=df
file=""
for element in form.split(","):
    value=element.split("=")
    file=file+'_'+value[1]
file=file[1 :].replace(":","")
path="Modeles/"+file+"_"+measurement
if  os.path.isdir(path) :
    Future=Existing_Predictor(df,host,measurement,look_back,"mse",1,10,1,form.split(","),freq_period=7)
    prediction,lower,upper=Future.make_prediction(len_prediction)
else : 
    model=train_linear_model(df,86,25)
    prediction=model.predict(np.linspace(len(df),len(df)+len_prediction-1,len_prediction).reshape(-1, 1)).reshape(1,-1)
    upper=prediction
    lower=prediction
dates=pd.date_range(df["ds"][len(df)-12],freq=time_obj.pandas_time, periods=len_prediction)
df2=pd.DataFrame({"ds" : dates , "yhat" : np.transpose(prediction[0]),"yhat_upper" :np.transpose(upper[0]),"yhat_lower" :np.transpose(lower[0]) })   
write_predictions(df2,client_2,measurement,host,db,form)


##### à supprimer 
cp = df_a[['ds', 'y']].copy()
cp["measurement"]="historical_"+measurement
client_2.delete_series(database=db,measurement="historical_"+measurement,tags={ "tag" : form[1 :]})
lines = [str(cp["measurement"][d]) 
         + ",type=forecast," 
         + form
         + " " 
         + "real_y=" + str(cp["y"][d]) + " " + str(int(time.mktime(datetime.strptime(str(cp['ds'][d]), "%Y-%m-%d %H:%M:%S").timetuple()))) + "000000000" for d in range(len(cp))]

client_2.write(lines,{'db':"telegraf"},204,'line')

########## fin partie à supprimer 
    
alert=Alert(host,measurement)
print(form)
alert.create('Attention le modèle associé à '+form.replace(","," AND ")+' dévie par rapport à la mémoire de : {{ index .Fields "performance_error" }}',","+form,period,typo,field)
alert.save_alert()
alert.launch()  
