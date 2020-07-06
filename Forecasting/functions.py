import pandas as pd
import time
import numpy as np
from datetime import datetime
from datetime import datetime, timedelta
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from Query_3 import Query_all
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import normaltest
from sklearn.linear_model import LinearRegression
from piecewise import piecewise

def modifie_df_for_fb(dfa,typo):
    if type(dfa)==list :
        df=dfa[0]
        TIME=[None]*len(df['time'])
    #modification du timestamp en format compréhensible par le modèle
        for i in range(len(df['time'])):
            dobj = datetime.strptime(df['time'][i], '%Y-%m-%dT%H:%M:%SZ')
            dobj.replace(tzinfo=None)
            dobj=dobj+ timedelta(hours=2)
            dobj= dobj.strftime('%Y-%m-%d %H:%M:%S')

            TIME[i]=dobj

        for i in range (len(dfa)):
            dfa[i]["time"]=TIME
            dfa[i]=dfa[i].rename(columns={"time":"ds", typo :"y"})
    else :
        df=dfa
        TIME=[None]*len(df['time'])
        #modification du timestamp en format compréhensible par le modèle
        for i in range(len(df['time'])):
            dobj = datetime.strptime(df['time'][i], '%Y-%m-%dT%H:%M:%SZ')
            dobj.replace(tzinfo=None)
            dobj=dobj+ timedelta(hours=2)
            dobj= dobj.strftime('%Y-%m-%d %H:%M:%S')

            TIME[i]=dobj

        df["time"]=TIME
        df=df.reset_index(drop=True)
        dfa=df.rename(columns={"time":"ds",typo :"y"})
    return dfa


def make_sliced_request(host, db, measurement, period, gb, past, typo, dic, field):
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
    cond = ""
    query = Query_all(db, host, typo, field, measurement)
    week = np.linspace(0, past, past + 1)
    li = [None] * len(week)
    for k in range(len(week) - 1):
        every = " now() -" + str(int(week[k + 1])) + 'w AND "time" < now() - ' + str(int(week[k])) + 'w '
        result, client = query.request(every, period, cond, gb)
        li[len(week) - 1 - k] = df
    df = pd.concat(li, axis=0, join="outer")
    df = df.reset_index()
    lli = df[["time", typo]]
    return lli, client


def make_sliced_request_multicondition(host, db, measurement, period, gb, cond, past, typo, dic, field):
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
    try:
        query = Query_all(db, host, typo, field, measurement)
        week = np.linspace(0, past, past + 1)
        li = [None] * len(week)
        for k in range(len(week) - 1):
            every = " now() -" + str(int(week[k + 1])) + 'w AND "time" < now() - ' + str(int(week[k])) + 'w '
            df, client = query.request(every, period, cond, gb)
            li[len(week) - 1 - k] = df
        df = pd.concat(li, axis=0, join="outer")
        df = df.reset_index()
        lli = df[["time", typo]]
        return lli, client
    except UnboundLocalError:
        if UnboundLocalError:
            st.write("Choisir l'ensemble des paramètres")


def write_predictions(df,client,measurement,host,db,form):
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
    if form[0] != "," :
        form=","+form
    print(df)
    df["measurement"]="ATT_PRED_"+measurement
    fill=(form[1 :].replace("=",",")).split(",")
    if len(form) < 2 :
        fill = "host="+host
        form = "," + fill
    key=[elt for idx, elt in enumerate(fill) if idx % 2 == 0]
    values=[elt for idx, elt in enumerate(fill) if idx % 2 != 0]
    dic=dict(zip(key,values))
    client.delete_series(database=db,measurement="ATT_PRED_"+measurement,tags=dic)
    cp = df[['ds', 'yhat','yhat_lower','yhat_upper','measurement']].copy()
    lines = [str(cp["measurement"][d])
                     + ",type=forecast"
                     + form
                     + " "
                     + "yhat=" + str(cp["yhat"][d]) + ","
                     + "yhat_lower=" + str(cp["yhat_lower"][d]) + ","
                     + "yhat_upper=" + str(cp["yhat_upper"][d])+ " " + str(int(time.mktime(datetime.strptime(str(cp['ds'][d]), "%Y-%m-%d %H:%M:%S").timetuple()))) + "000000000" for d in range(len(cp))]
    try :
        client.write(lines,{'db':db},204,'line')
    except Exception :
        print("probblems when sending some values")



from sklearn.base import BaseEstimator, TransformerMixin

class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, input_array, y=None):
        return self
    
    def transform(self, input_array, y=None):
        return input_array*1
    def inverse_transform(self,input_array,y=None):
        return input_array*1



def make_form(df,host):
    '''
    This function change form to adapt with the granularity of the data received
    Parameters
    ----------
    df : dataframe
        data .

    Returns
    -------
    form : str

    '''
    columns=df.columns
    df=df.reset_index()
    form=""
    for j in range(len(columns)):                     # utilise les colonnes pour pouvoir adapter l'ecriture dans la table
        if columns[j] != "ds" and columns [j] != "mean" and columns[j] != "index" and columns[j] != "y"  :
                form=form+","+columns[j]+"="+str(df[columns[j]][0])
    print(form)
    if len(form) < 1:
        form =",host="+host
    return form

def evaluate_linearity(df):
    '''
    This fucntion compute the autocorrelation function of the signal and then caracterize how linear this autocorrelation is.(Minus the fist point)

    Parameters
    ----------
    df : dataframe
        data.

    Returns
    -------
    r_sq : int
        Pearson's coefficient.

    '''
    tab=[df["y"].autocorr(lag) for lag in range(2,60)]
    x=np.linspace(1,len(tab)-1,len(tab))
    model = LinearRegression()
    model.fit(x.reshape(-1,1),np.asarray(tab).reshape(-1,1))
    r_sq=model.score(x.reshape(-1,1), np.asarray(tab).reshape(-1,1))
    return r_sq

def train_linear_model(df,borne_max,severity):
    '''
    Determine the last linear part of data to concider and then train the model of this subsequence of data.

    Parameters
    ----------
    df : dataframe
        data.

    Returns
    -------
    model : linear_model
        linear model trained on the last trend of the signal.
    r_sq : int
        Pearson's coefficient (caracterize how well the model represent the data).
    df_l : dataframe
        last linear part of the data.

    '''
    x=np.arange(len(df["y"]-1))
    model = piecewise(x,df["y"])
    #piecewise_plot(x, df["y"], model=model)
    """dx = 1
    dy=signal.savgol_filter(df["y"],24*7+1,1)
    #dy = diff(dy)/dx
    #dy=df["y"].rolling(24).mean()
    #dy = diff(dy)/dx
    plt.plot(dy)
    i=1
    mini=np.amin(dy)
    maxi=np.amax(dy)
    j=0
    nb_peaks = len(df["y"])+1
    while nb_peaks > len(df["y"])/10 :
        peaks, _ = find_peaks(dy, height=(10+j)*(maxi-mini)/100)
        nb_peaks=len(peaks)
        j=j+1
        plt.plot(peaks, abs(dy[peaks]), "x")
        plt.plot(abs(dy),label="abs(dy)")
        plt.plot(df["y"],label="df['y']")
        plt.title("Slope changes")
        plt.legend()
        plt.show()
    plt.plot(peaks,df["y"][peaks])
    dy = diff(df["y"][peaks])/dx
    plt.plot(dy)
    plt.show()
    print("hbefiefizbin",peaks)
    plt.plot(abs(dy))
    plt.plot(abs(peaks[-1]+1), "x")
    plt.show()"""

    """while abs(dy[len(dy)-i-1]) < severity*(maxi-mini)/100 or i < 2 :
        i=i+1
    if i > borne_max :
        i=borne_max
    i=peaks[-1]+1
    print("la longueur est ",len(df))
    print("le sous segment est ",i+1)
    df_l=df["y"][i+1 :]
    #plt.plot(df_l)
    #plt.show()

    #plt.plot(df["y"][i+1 :])
    #plt.show()
    x=np.linspace(1,len(df_l)-1,len(df_l))
    model = LinearRegression()
    model.fit(x.reshape(-1,1),np.asarray(df_l).reshape(-1,1))
    r_sq=model.score(x.reshape(-1,1), np.asarray(df_l).reshape(-1,1))"""
    return model


def decoupe_dataframe(df,look_back):
    dataX,dataY = [],[]
    for i in range(len(df) - look_back - 1):
        a = df[i:(i + look_back)]
        dataY=dataY+[df[i+look_back]]
        dataX.append(a)
    return (np.asarray(dataX),np.asarray(dataY).flatten())

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
