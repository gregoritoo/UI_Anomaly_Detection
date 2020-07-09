
import os
import numpy as np
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import SeasonalAD
from adtk.detector import InterQuartileRangeAD
from sklearn.linear_model import LinearRegression
from adtk.detector import RegressionAD
from adtk.detector import PersistAD
from adtk.detector import LevelShiftAD
from adtk.detector import VolatilityShiftAD
from adtk.detector import AutoregressionAD
from influxdb import InfluxDBClient
import pandas as pd
import matplotlib.pyplot as plt
from Functions.ml_functions import mad_numpy
import plotly.graph_objects as go
import streamlit as st

INFLUX_NAME= os.environ['INFLUX_NAME']
INFLUX_PASSWORD= os.environ['INFLUX_PASSWORD']
INFLUX_PORT= 443
INFLUX_HOST= "influxdb-monitoring.supralog.com"
INFLUX_SSL= True
INFLUX_VERIFY_SSL= True

def decoupe_dataframe(df,look_back):
    dataX,dataY = [],[]
    for i in range(len(df) - look_back - 1):
        a = df[i:(i + look_back)]
        dataY=dataY+[df[i+look_back]]
        dataX.append(a)
    return (np.asarray(dataX),np.asarray(dataY).flatten())



def connect_database(db):
    client = InfluxDBClient(host=INFLUX_HOST, port=INFLUX_PORT, username=INFLUX_NAME, password=INFLUX_PASSWORD, ssl=INFLUX_SSL, verify_ssl=INFLUX_VERIFY_SSL)
    client.switch_database(db)
    client=client
    return client

def get_field_names(results,measurement):
    P = []
    for i in range(len(results.raw["series"])):
        j = 0
        if results.raw["series"][i]["name"] == measurement:
            if type(results.raw["series"][i]["values"][:]) != int:
                P = P + results.raw["series"][i]["values"]
    return np.array(P)


def select_apply_model(Model,df,dfa_2,period,host,measurement,path,form):
    df=df.set_index("ds")
    df.index=pd.to_datetime(df.index)
    s=df["y"]
    s = validate_series(s)

    dfa_2=dfa_2.set_index("ds")
    dfa_2.index=pd.to_datetime(dfa_2.index)
    s2=dfa_2["y"]
    s2 = validate_series(s2)

    if Model == "SeasonalAD" :
        st.write("INDICATION : Ce modèle est adapté aux données qui présentent une seasonalité ie les mêmes motifs se répétent à la même periode de la semaine et de la journée ")
        c = st.number_input('Select coefficient (default 1.5) ')
        st.write("INDICATION : 'C' est la tolérance du modèle au variation par rapport à la normal ie plus 'C' augmente et plus le nombre d'anomalies diminue ")
        model=SeasonalAD(c=c)
        model=do_anomaly_detection(s,model,"marker",s2)

    elif Model == "InterQuartileRangeAD":
        st.write("INDICATION : Ce modèle detecte les points extrêmes")
        st.write( "INDICATION :  'C' est la variance de la loi normale centré réduite , plus 'C' augmente plus la probabilité de considérer un point comme normal augmente : (à 1.96 99% des points d'entrainement sont considérés comme normaux ; à 1.5 c'est 95% et à 1 c'est 67% )")
        c = st.number_input('Select coefficient (default 1.5)')
        model = InterQuartileRangeAD(c=c)
        model = do_anomaly_detection(s,model,"marker",s2)

    elif Model == "PersistAD":
        st.write("INDICATION : Ce modèle detecte les changement de nature d'une série temporelle passant de volatile à constante")
        st.write("INDICATION : Facteur des ecarts acceptés comme non anomalies. Ce facteur est basé sur l'écart inter quartile de l'historique de données donc plus la distribution des données est large plus les variations acceptées seront grandes. En augmentant le facteur on augmente la tolérence du modèle au grand écart ")
        c = st.number_input('Select factor (default 3) ')
        model = PersistAD(c=c, side='positive')
        model = do_anomaly_detection(s,model,"span",s2)

    elif Model == "LevelShiftAD":
        st.write("INDICATION : Ce modèle detecte les changement de paliers dans les données. Peut-être utile pour les données de type disk ou mém")
        st.write("INDICATION : Facteur des ecarts acceptés comme non anomalies. Ce facteur est basé sur l'écart inter quartile de l'historique de données donc plus la distribution des données est large plus les variations acceptées seront grandes. En augmentant le facteur on augmente la tolérence du modèle au grand écart ")
        c = st.number_input('Select factor (default 6) ')
        option = st.sidebar.selectbox('Level shift on which side ?',["both","positive","negative"])
        model = LevelShiftAD(c=c, side=option, window=5)
        model = do_anomaly_detection(s,model,"span",s2)


    elif Model == "VolatilityShiftAD":
        st.write( "INDICATION : Ce modèle  detecte les changements de nature d'une série temporelle de linaire à volatille ")
        c = st.number_input('Select factor (default 6) ')
        model = VolatilityShiftAD(c=c, side='positive', window=30)
        model =do_anomaly_detection(s,model,"span",s2)

    elif Model == "AutoregressionAD":
        st.write("Ce modèle utilise une combinaison linéaire des points passés pour prédire le future. Ce modèle est optimal pour des données présentants un pattern avec quelques variations")
        nb_steps = st.number_input('Select order of AR model (default 7*2) ')
        st.write("L'odre du modèle AR correspond aux nombres de points utlisée pour décrire le points suivant ex AR d'ordre 1 n'utilise que le point N-1 pour décrire le point N. Ce modèle est a privilégié si les données sont cyclique mais pas seasonal ie se répétent mais pas formcément au même moment de la semaine etc)")
        step_size = st.number_input('Select step sier  (default 24) ')
        st.write("Nombre de pas entre chaque point ")
        c=st.number_input("Select severity of the model (default 3)")
        model = AutoregressionAD(n_steps=int(nb_steps), step_size=int(step_size), c=c)
        model = do_anomaly_detection(s,model,"marker",s2)

    elif Model == "Modele_custom":
        c = st.number_input('Select severity ')
        w = st.number_input('Select window size ')
        apply_custom_model(df, dfa_2, c, int(w))

    elif Model=="Model_VAR_LSTM":
        c = st.number_input('Select C ')
        butt = st.button("Retrain anomaly detection modele")
        model=apply_VAR_LSTM(form, measurement, host, df, dfa_2,int(c),s,s2,butt)

    elif Model=="Change_in_relation" :
        regression_ad = RegressionAD(regressor=LinearRegression(), target="Speed (kRPM)", c=3.0)
        anomalies = regression_ad.fit_detect(df)
        plot(df, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=0.3,
             curve_group='all')
    return model,c


def apply_VAR_LSTM(form,measurement,host,df,dfa_2,c,s,s2,butt):
    '''
        This function creates the model if it doesn't exist or apply it to historical data otherweise
    :param form: str string that represents the series key in line protocol ex  host=host1,cpu=cpu-total
    :param measurement: str Influxdb measurement
    :param host: str Host of interest
    :param df: dataframe of last period
    :param dfa_2: dataframe of period of interest (default Sep 2019)
    :param c: severity of the model
    :param s: panda series containing df numerical values
    :param s2: panda series containing dfa_2 numerical values
    :param butt: boolean become True when the user trig the "train encoder model" button on the interface
    :return:
    '''
    file = ""
    if type(form) != list:
        form = form[1:].split(",")
    else:
        form = form
    try:
        for element in form:
            value = element.split("=")
            file = file + '_' + value[1]
        file = file[1:].replace(":", "")
    except Exception:
        file = host
    file = file.replace(" ", "")
    path = "Modeles_AD/" + file + "_" + measurement
    if not os.path.isdir(path):
        os.makedirs(path)
    path = "Modeles_AD/" + file + "_" + measurement + "/" + "var" + ".h5"
    print(path)
    from var_encoder import New_VAR_LSTM, Existing_VAR_LSTM
    if not os.path.isfile(path) or butt :
        var_encoder = New_VAR_LSTM(c, c, df, form, measurement, host)
        model = 0
    else:

        var_encoder = Existing_VAR_LSTM(c, path, form, measurement, host)
        model, scaler = var_encoder.load_models()
        prediction = var_encoder.make_prediction(df["y"], model, c, scaler)
        prediction = np.append(prediction[0], [prediction[i][-1] for i in range(1, len(prediction))])
        pplot(s,pd.Series(prediction))
        prediction = var_encoder.make_prediction(dfa_2["y"], model, c, scaler)
        prediction = np.append(prediction[0], [prediction[i][-1] for i in range(1, len(prediction))])
        pplot(s2, pd.Series(prediction))
    return model


def apply_custom_model(df,dfa_2,c,w):
    '''
        This function frames the time series by MAD boudries (not implemented totaly yet)
    :param df: dataframe of last period
    :param dfa_2: dataframe of period of interest (default Sep 2019)
    :param c: int severity of the model
    :param w: int size of window
    :return:
    '''
    #c = 5
    #w = 5
    median = np.array(df.rolling(w).median().dropna())
    std = np.array(df["y"])
    std = mad_numpy(std, w)
    print(len(median[:, 0] - c * std[:]))
    print(len(median[:, 0] + c * std[:], ))
    x = np.arange(0, len(median[:, 0] + c * std[:]), 1)
    plt.fill_between(x + w - 1, median[:, 0] + c * std[:], median[:, 0] - c * std[:], color="blue")
    for i in range(w - 1, len(df["y"])):
        if abs(df["y"][i - w + 1] - median[i - w + 1]) > c * std[i - w + 1]:
            plt.scatter(i - w + 1, df["y"][i - w + 1], marker="x", color="black")
    plt.plot(np.array(df["y"]), color="red", label="real value")
    st.pyplot()
    median = np.array(dfa_2.rolling(w).median().dropna())
    std = np.array(dfa_2["y"])
    std = mad_numpy(std, w)
    print(len(median[:, 0] - c * std[:]))
    print(len(median[:, 0] + c * std[:], ))
    x = np.arange(0, len(median[:, 0] + c * std[:]), 1)
    plt.fill_between(x, median[:, 0] + c * std[:], median[:, 0] - c * std[:], color="blue")
    for i in range(w - 1, len(dfa_2["y"])):
        if abs(dfa_2["y"][i - w + 1] - median[i - w + 1]) > c * std[i - w + 1]:
            plt.scatter(i - w + 1, dfa_2["y"][i - w + 1], marker="x", color="black")
    plt.plot(np.array(dfa_2["y"]), color="red", label="real value")
    st.pyplot()

def do_anomaly_detection(s,model,marker,s2):
    '''
        This function apply the model to the historical data and plot the results
    :param s: time series
    :param model: adtk model for anomaly detection
    :param marker: maker is either "marker" or "span" for the plot
    :return:
        model
    '''
    anomalies = model.fit_detect(s)
    pplot(s,anomalies)
    anomalies_2 = model.predict(s2)
    pplot(s2, anomalies_2)
    return model




def pplot(s,anomalies):
    '''
        This function plot the time serie with the abnormalous points in red. In order to keep the duration as short as possible it does not plot the figure if there is more than 4000 abnomalous points.
    :param s: panda.series with points of interrest
    :param anomalies: array with either bool or binary value (1 or True if the point is a anomaly, 0 or False otherweise
    :return:
    '''
    if len(anomalies) < 4000 :
        df=pd.DataFrame()
        df["date"]=s.index
        df['y']=s.values
        fig = go.Figure()
        fig.add_scatter( x=df['date'], y=df['y'])
        for i in range(len(anomalies)) :
            if anomalies.values[i] == 1 or anomalies.values[i] == True :
                fig.add_scatter(x=[df["date"][i]],y=[df["y"][i]],mode='markers',name=str(df["date"][i]),marker_color="red")
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig)
    else :
        st.write("Trop de points anormaux pour pouvoir être affichés")


def contextual_fields(measurement,key,cond,gb,client,num,analyse=False):
    '''
        This function exctract tag key and tag value and make the interactive interface accordingly
    :param measurement:
        Influx measurement
    :param key:
        key tag which caraterize the database
    :param cond:
        condition string ; each condition is separate by AND
    :param gb:
        group by condtion , each element is separate by a coma
    :return:
        cond :condition string updated
        field : selected key value by the user on the UI
        results : array of possible key values
        gb : groupby string updated
    '''

    if len(key) > 2 :
        if analyse == False :
            results = client.query('SHOW TAG VALUES WITH KEY = "'+key+'"')
            P = get_field_names(results, measurement)
            choice= "Choose "+ key
            condition = st.sidebar.selectbox(
                choice +str(num),
                P[:, 1]
            )
        else :
            results = client.query('SHOW TAG VALUES WITH KEY = "' + key + '"')
            P = get_field_names(results, measurement)
            choice = "Choose " + key
            condition = st.sidebar.multiselect(
                choice,
                P[:, 1]
            )
        results = client.query('SHOW FIELD KEYS')
        P = get_field_names(results, measurement)
        field = st.sidebar.selectbox(
            " Choose field "+str(num),
            P[:, 0]
        )
        if analyse :
            cond = [cond + " AND " + key + "='" + condi + "' " for condi in condition]
        else :
            cond = cond + " AND " + key + "='" + condition + "' "
        gb = gb + key + ","

        return cond, field, results, gb
    else :
        results = client.query('SHOW FIELD KEYS')
        P = get_field_names(results, measurement)
        field = st.sidebar.selectbox(
            " Choose field "+str(num),
            P[:, 0]
        )
        cond= cond+""
    return cond,field,results,gb



measurement_field = {
    "cpu": "cpu",
    "disk": "device",
    "mem" : "",
    "ga_page_tracking" : "viewName",
    "ga_real_time" : "report",
    "accesLogs":"client",
    "ga_sites_users" : "viewName",
    "mandrill_email_usage" : "api_key_name",
    "healthCheck" : "service",
    "appErrors" : "client",
    "ga_site_speed" :  "viewName" ,
    "net" : "",
    "http_response" : "",
    "kernel" : "",
}



def write_request(measurement,cond,gb,client,num,analyse):
    '''

    :param measurement: str influxdb'measurement
    :param cond: str string containing all the condition in INFLUXSQL like syntax
    :param gb: str string containing all the groupby in INFLUXSQL like syntax
    :param client: InfluxdbClient
    :param int:  destinguish selectboxes if there is multiple choice
    :param analyse: Boolean destinguish analyse (page 2) because needs multiplechoicebox instead of selectbox
    :return:
        updated cond
        field Selected influxdb field of interest
        results list of kay values
        updated gb
    '''
    cond,field,results,gb=contextual_fields(measurement,measurement_field[measurement],cond,gb,client,num,analyse)
    return cond,field,results,gb




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


