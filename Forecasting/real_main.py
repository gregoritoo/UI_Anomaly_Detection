

import pandas as pd
import time
import numpy as np
import pickle
from datetime import datetime
from Alert import Alert
from datetime import datetime, timedelta
from prediction_functions import write_predictions,modifie_df_for_fb,make_sliced_request,evaluate_linearity,train_linear_model
from influxdb import InfluxDBClient
from prediction_functions import make_sliced_request,modifie_df_for_fb,make_form,transform_time
from statsmodels.tsa.seasonal import seasonal_decompose
from New_Predictor import New_Predictor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from  scipy import stats
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from numpy import diff


def predict(df,form,len_prediction,host,measurement,db,severity,freq_period,look_back):
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
    
period="60m"
time_obj=Time(period)
typo="mean"
db="telegraf"
measurements=["mem"]#","disk","mem"]
hosts=["PROD-SGDF","QUAL-SGDF","DB-SGDF","PROD-INTRA-MUT4","DB-INTRASSOC","DBINTRASSOC2016","PROD-INTRA-MUT1","PROD-INTRA-MUT2","PROD-INTRA-MUT3"]
gb="host,"
severity=2
freq_period=24+1   #doit faire une semaine en générale suivant la périodicité supposé du signal(à calculer en fonvtion de la féquence d'échantillonage)
 #########a supprimer avant de mettre en prod

client_2 = InfluxDBClient(host="localhost", port=8086)
client_2.switch_database(db)
########## fin partie à supprimer 

len_prediction=transform_time(period)* 12
nb_week_to_query=12
look_back = transform_time(period)
for measurement in measurements :
    for host in hosts :
        df,client=make_sliced_request(host,db,measurement,period,gb,nb_week_to_query,typo)
        dfl=modifie_df_for_fb(df,typo)             #la requete retourne une liste de dataframe de taille  : multiplication des elements du groupby entre eux ex gb host,device = nbhost*nbdevice 
        if type(df) != list :
            df=dfl
            if df["y"].isna().sum() > transform_time(period)*4 :    # pas assez de données pour pouvoir créer un modèle 
                   break 
            form=make_form(df,host)
            df_a=df
            prediction,lower,upper=predict(df,form,len_prediction,host,measurement,db,severity,freq_period,look_back)
            print(df["ds"][len(df)-1])
            dates=pd.date_range(df["ds"][len(df)-1],freq=time_obj.pandas_time, periods=len_prediction)
            df2=pd.DataFrame({"ds" : dates , "yhat" : np.transpose(prediction[0]),"yhat_upper" :np.transpose(upper[0]),"yhat_lower" :np.transpose(lower[0]) })
            df=df.append(df2[ :]).reset_index()    
            write_predictions(df,client_2,measurement,host,db,form)
            #########a supprimer avant de mettre en prod
            cp = df_a[['ds', 'y']].copy()
            cp["measurement"]="mean_y2"+measurement
            client_2.delete_series(database=db,measurement="mean_y2"+measurement,tags={ "tag" : form[1 :]})
            lines = [str(cp["measurement"][d]) 
                     + ",type=forecast"
                     + form
                     + " " 
                     + "real_y=" + str(cp["y"][d]) + " " + str(int(time.mktime(datetime.strptime(str(cp['ds'][d]), "%Y-%m-%d %H:%M:%S").timetuple()))) + "000000000" for d in range(len(cp))]
            try :
                client_2.write(lines,{'db':db},204,'line')
            except Exception :
                print("pb")
            ########## fin partie à supprimer 
            alert=Alert(host,measurement)
            alert.create('Attention le modèle associé à '+form.replace(","," AND ")+' dévie par rapport à la mémoire de : {{ index .Fields "performance_error" }}',form,period)
            alert.save_alert()
            alert.launch()  
        elif type(df) == list  :
            for i in range(len(df)) :
                df=dfl[i].dropna()
                if len(df) >  transform_time(period)*4 :    # pas assez de données pour pouvoir créer un modèle 
                    form=make_form(df,host)
                    df_a=df
                    prediction,lower,upper=predict(df,form,len_prediction,host,measurement,db,severity,freq_period,look_back)
                    dates=pd.date_range(df["ds"][len(df)-1],freq=time_obj.pandas_time, periods=len_prediction)
                    df2=pd.DataFrame({"ds" : dates , "yhat" : np.transpose(prediction[0]),"yhat_upper" :np.transpose(upper[0]),"yhat_lower" :np.transpose(lower[0]) })
                    df=df.append(df2[ :]).reset_index()    
                    write_predictions(df,client_2,measurement,host,db,form)
                    ########### a supprimer avant de mettre en prod
                    df_a=df_a.reset_index()
                    cp =  df_a[['ds', 'y']].copy()
                    cp["measurement"]="mean_y2"+measurement
                    client_2.delete_series(database=db,measurement="mean_y2"+measurement,tags={ "tag" : form[1 :]})
                    lines = [str(cp["measurement"][d]) 
                             + ",type=forecast"
                             + form
                             + " " 
                             + "y=" + str(cp["y"][d]) + " " + str(int(time.mktime(datetime.strptime(str(cp['ds'][d]), "%Y-%m-%d %H:%M:%S").timetuple()))) + "000000000" for d in range(len(cp))]
                    try :
                        client_2.write(lines,{'db':db},204,'line')
                    except Exception :
                        print("pb")   
                    ############ fin a supprimer avant de mettre en prod
                    alert=Alert(host,measurement)                  
                    message='Attention le modèle associé à '+form[1 :].replace(","," AND ")+' dévie par rapport à la mémoire de : {{ index .Fields "performance_error" }}'
                    alert.create(message,form,period)
                    alert.save_alert()
                    alert.launch()
        elif df == -1 :
            print("Influx returns no values, verifie the request parameters")
                    

           
