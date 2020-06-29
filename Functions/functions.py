# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:05:20 2020

@author: GSCA
"""


import pandas as pd
import time
import numpy as np
from datetime import datetime
from query import query
from datetime import datetime, timedelta
from pandas import Series
from sklearn.preprocessing import MinMaxScaler 
from Query_2 import Query_Mem,Query_Disk,Query_Cpu
import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import normaltest
from scipy.stats import shapiro

from scipy.stats import anderson
def modifie_df_for_fb(df,typo):
    TIME=[None]*len(df['time'])
    DAY=[None]*len(df['time'])
    HOUR=[None]*len(df['time'])
    #modification du timestamp en format compréhensible par le modèle
    for i in range(len(df['time'])):
        dobj = datetime.strptime(df['time'][i], '%Y-%m-%dT%H:%M:%SZ')
        dobj.replace(tzinfo=None)
        dobj2=datetime.strptime(df['time'][i], '%Y-%m-%dT%H:%M:%SZ')
        dobj3=datetime.strptime(df['time'][i], '%Y-%m-%dT%H:%M:%SZ')
        dobj2=dobj2.strftime('%A')
        dobj3=dobj3.strftime('%H')
        dobj=dobj+ timedelta(hours=2)
        dobj= dobj.strftime('%Y-%m-%d %H:%M:%S')
       
        TIME[i]=dobj
        DAY[i] = dobj2
        HOUR[i] = dobj3
    df["time"]=TIME
    df["day"]=DAY
    df["hour"]=HOUR
    df=df.reset_index(drop=True)
    #suppression des colonnes inutiles
    df=df.rename(columns={"time":"ds", typo:"y"})
    return df

def make_sliced_request(host,db,measurement,period,gb):    
    '''
    This function slices the request on requests of one week length and after that joins all the data in one dataframe
    
    Parameters
    ----------
    host : str
        Influxdb host we want to work on.
    db : str
        influxdb database to connect to.
    measurement : str
        Influxdb's measurement .
    period : str
        windows to focus on (duration in influxdb syntax ix "5m").
    gb : str
        group by condition of the requeste to write with influx db syntax
        ex "host,element_to_groupby," => always end with ","
      
    Returns
    -------
    df : str
        dataframe contening all the data.
    host : str
        name of the host, not useful.
    client : Influxdb object
        API to interact with the database, will be needed to write the prediction later.
    '''
    if measurement=="mem" :
        query=Query_Mem(db)
    elif measurement =="cpu":
        query=Query_Cpu(db)
    elif measurement =="disk":
       query=Query_Disk(db)
    cond="AND host='"+host+"'"
    week=np.linspace(0,12,13)
    li=[None]*len(week)
    for k in  range (len(week)-1) :
        every=str(int(week[k+1])) + 'w AND "time" < now() - '+ str(int(week[k])) + 'w ' 
        result,client =query.request(every,period,cond,gb)
        df=result[host]
        li[len(week)-1-k]=df
    df= pd.concat(li, axis=0, join="outer")
    df=df.reset_index()
    df=df[["time","mean"]]
    return df,client

def write_predictions(df,client,measurement,host,db):
    '''
    This function write  predictions in the proper way to be send to the database and then send it.
    
    Parameters
    ----------
    df : dataframe
        no needed .
    client : Influxdb object
        DESCRIPTION.
    measurement : str
        Influxdb's measurement .
    host : str
        name of thz host, not useful.
    forecast : dataframe
        extended dataframe with the prediction included.
    db : str
        influxdb database to connect to.
                
    Returns
    -------
    None.
    '''  
    df["measurement"]="pred2_"+measurement
    client.delete_series(database=db,measurement="pred2_"+measurement,tags={ "host" : host})
    cp = df[['ds', 'yhat','yhat_lower','yhat_upper','measurement']].copy()

    lines = [str(cp["measurement"][d]) 
                     + ",type=forecast"
                     +",host="+host
                     + " " 
                     + "yhat=" + str(cp["yhat"][d]) + ","
                     + "yhat_lower=" + str(cp["yhat_lower"][d]) + ","
                     + "yhat_upper=" + str(cp["yhat_upper"][d])+ " " + str(int(time.mktime(datetime.strptime(str(cp['ds'][d]), "%Y-%m-%d %H:%M:%S").timetuple()))) + "000000000" for d in range(len(cp))]
    try :
        client.write(lines,{'db':db},204,'line')
    except Exception :
        print("probblems when sending some values")
        




def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


        

        
def one_week_data_request(host,db,measurement,period,gb):
    if measurement=="mem" :
        query=Query_Mem(db)
    elif measurement =="cpu":
        query=Query_Cpu(db)
    elif measurement =="disk":
       query=Query_Disk(db)
    cond="AND host='"+host+"'"
    every=str(1) + 'w '
    result,client =query.request(every,period,cond,gb)
    df = result[host]
    df.reset_index()
    return df,client
