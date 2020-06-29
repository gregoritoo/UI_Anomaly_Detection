from influxdb import InfluxDBClient
import os
import numpy as np
import streamlit as st
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
from Existing_Predictor import Existing_Predictor

INFLUX_NAME= os.environ['INFLUX_NAME']
INFLUX_PASSWORD= os.environ['INFLUX_PASSWORD']
INFLUX_PORT= 443
INFLUX_HOST= "influxdb-monitoring.supralog.com"
INFLUX_SSL= True
INFLUX_VERIFY_SSL= True





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
        do_anomaly_detection(s,model,"marker",s2)

    elif Model == "InterQuartileRangeAD":
        st.write("INDICATION : Ce modèle detecte les points extrêmes")
        st.write( "INDICATION :  'C' est la variance de la loi normale centré réduite , plus 'C' augmente plus la probabilité de considérer un point comme normal augmente : (à 1.96 99% des points d'entrainement sont considérés comme normaux ; à 1.5 c'est 95% et à 1 c'est 67% )")
        c = st.number_input('Select coefficient (default 1.5)')
        model = InterQuartileRangeAD(c=c)
        do_anomaly_detection(s,model,"marker",s2)

    elif Model == "PersistAD":
        st.write("INDICATION : Ce modèle detecte les changement de nature d'une série temporelle passant de volatile à constante")
        st.write("INDICATION : Facteur des ecarts acceptés comme non anomalies. Ce facteur est basé sur l'écart inter quartile de l'historique de données donc plus la distribution des données est large plus les variations acceptées seront grandes. En augmentant le facteur on augmente la tolérence du modèle au grand écart ")
        c = st.number_input('Select factor (default 3) ')
        model = PersistAD(c=c, side='positive')
        do_anomaly_detection(s,model,"span",s2)

    elif Model == "LevelShiftAD":
        st.write("INDICATION : Ce modèle detecte les changement de paliers dans les données. Peut-être utile pour les données de type disk ou mém")
        st.write("INDICATION : Facteur des ecarts acceptés comme non anomalies. Ce facteur est basé sur l'écart inter quartile de l'historique de données donc plus la distribution des données est large plus les variations acceptées seront grandes. En augmentant le facteur on augmente la tolérence du modèle au grand écart ")
        c = st.number_input('Select factor (default 6) ')
        option = st.sidebar.selectbox('Level shift on which side ?',["both","positive","negative"])
        model = LevelShiftAD(c=c, side=option, window=5)
        do_anomaly_detection(s,model,"span",s2)


    elif Model == "VolatilityShiftAD":
        st.write( "INDICATION : Ce modèle  detecte les changements de nature d'une série temporelle de linaire à volatille ")
        c = st.number_input('Select factor (default 6) ')
        model = VolatilityShiftAD(c=c, side='positive', window=30)
        do_anomaly_detection(s,model,"span",s2)

    elif Model == "AutoregressionAD":
        st.write("Ce modèle utilise une combinaison linéaire des points passés pour prédire le future. Ce modèle est optimal pour des données présentants un pattern avec quelques variations")
        nb_steps = st.number_input('Select order of AR model (default 7*2) ')
        st.write("L'odre du modèle AR correspond aux nombres de points utlisée pour décrire le points suivant ex AR d'ordre 1 n'utilise que le point N-1 pour décrire le point N. Ce modèle est a privilégié si les données sont cyclique mais pas seasonal ie se répétent mais pas formcément au même moment de la semaine etc)")
        step_size = st.number_input('Select step sier  (default 24) ')
        st.write("Nombre de pas entre chaque point ")
        model = AutoregressionAD(n_steps=7 * 2, step_size=24, c=3.0)
        do_anomaly_detection(s,model,"marker",s2)

    elif Model == "Model_IA":
        st.write("ATTENTION ! Le modèle utilisé est celui pour faire les prédictions donc la fréqeunce d'extraction des données doit être la même que celle choisit lors de l'entrainement initial,si la prévisualisation n'affiche rien augmenter le nombre de semaines à requêter ")
        c = st.number_input('Select C ')
        st.write("INDICATION : 'C' est l'ecart à partir duquel un point est concidéré comme une anomalie ")
        look_back = transform_time(period)*1
        taille_motif  = st.number_input("Taille du motif qui se répète ie nombre impaire qui correspond à la fréquence d'échantillonage")
        Future = Existing_Predictor(df, host, measurement,look_back , "mse", 1, 10, 1, form.split(","), freq_period=int(taille_motif),file=path)
        Future.make_prediction(df,c)
        st.pyplot()
        Future.make_prediction(dfa_2,c)
        plt.title("Comparaison avec la période sélectionnée ")
        st.pyplot()
    elif Model=="Change_in_relation" :
        regression_ad = RegressionAD(regressor=LinearRegression(), target="Speed (kRPM)", c=3.0)
        anomalies = regression_ad.fit_detect(df)
        plot(df, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=0.3,
             curve_group='all');
    if model :
        return model
    else :
        return c

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
    plot(s, anomaly=anomalies, ts_markersize=1, anomaly_color='red', anomaly_tag=marker, anomaly_markersize=10,
         figsize=(24, 12), ts_alpha=0.9, ts_linewidth=4)
    st.pyplot()
    anomalies_2 = model.predict(s2)
    plot(s2, anomaly=anomalies_2, ts_markersize=1, anomaly_color='red', anomaly_tag=marker, anomaly_markersize=10,
         figsize=(24, 12), ts_alpha=0.9, ts_linewidth=4)
    st.pyplot()
    return model


def contextual_fields(measurement,key,cond,gb,client,num):
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
        results = client.query('SHOW TAG VALUES WITH KEY = "'+key+'"')
        P = get_field_names(results, measurement)
        choice= "Choose "+key
        condition = st.sidebar.selectbox(
            choice +str(num),
            P[:, 1]
        )
        results = client.query('SHOW FIELD KEYS')
        P = get_field_names(results, measurement)
        field = st.sidebar.selectbox(
            " Choose field "+str(num),
            P[:, 0]
        )
        cond = cond + " AND " + key + "='" + condition + "' "
        gb = gb + key + ","
    else :
        results = client.query('SHOW FIELD KEYS')
        P = get_field_names(results, measurement)
        field = st.sidebar.selectbox(
            " Choose field "+str(num),
            P[:, 0]
        )
        cond= cond+""
    return cond,field,results,gb

def contextual_fields_analyse(measurement,key,condi,gb,client):
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
        results = client.query('SHOW TAG VALUES WITH KEY = "'+key+'"')
        P = get_field_names(results, measurement)
        choice= "Choose "+key
        condition = st.sidebar.multiselect(
            choice,
            P[:, 1]
        )
        cond = [condi + " AND " + key + "='" + cond + "' " for cond in condition]
        results = client.query('SHOW FIELD KEYS')
        P = get_field_names(results, measurement)
        field = st.sidebar.selectbox (
            " Choose field ",
            P[:, 0]
        )

        gb = gb + key + ","
    else :
        results = client.query('SHOW FIELD KEYS')
        P = get_field_names(results, measurement)
        field = st.sidebar.selectbox (
            " Choose field ",
            P[:, 0]
        )
        cond= cond+""
    return cond,field,results,gb


def write_request(measurement,cond,gb,client,num):
    if measurement == "cpu":
        cond,field,results,gb=contextual_fields(measurement,"cpu",cond,gb,client,num)

    elif measurement == "disk":
        cond, field, results, gb = contextual_fields(measurement, "device", cond, gb,client,num)


    elif measurement == "mem":
        cond, field, results, gb = contextual_fields(measurement, "", cond, gb,client,num)


    elif measurement == "ga_page_tracking" :
        cond, field, results, gb = contextual_fields(measurement, "viewName", cond, gb,client,num)

    elif measurement == "ga_real_time" :
        cond, field, results, gb = contextual_fields(measurement, "report", cond, gb,client,num)


    elif measurement == "accesLogs" :
        cond, field, results, gb = contextual_fields(measurement, "client", cond, gb,client,num)


    elif measurement == "ga_sites_users" :
        cond, field, results, gb = contextual_fields(measurement, "viewName", cond, gb,client,num)


    elif measurement == "mandrill_email_usage" :
        cond,field,results,gb=contextual_fields(measurement,"api_key_name",cond,gb,client,num)

    elif measurement == "healthCheck" :
        cond,field,results,gb=contextual_fields(measurement,"service",cond,gb,client,num)

    elif measurement == "appErrors" :
        cond,field,results,gb=contextual_fields(measurement,"client",cond,gb,client,num)

    elif measurement == "ga_site_speed" :
        cond,field,results,gb=contextual_fields(measurement,"viewName",cond,gb,client,num)

    elif measurement == "net" or  measurement == "http_response" or  measurement == "kernel":
        cond, field, results, gb = contextual_fields(measurement, "", cond, gb,client,num)


    else :
        cond=cond +""

    return cond,field,results,gb


def write_request_analyse(measurement,cond,gb,client,num):
    if measurement == "cpu":
        cond,field,results,gb=contextual_fields_analyse(measurement,"cpu",cond,gb,client)

    elif measurement == "disk":
        cond, field, results, gb = contextual_fields_analyse(measurement, "device", cond, gb,client)


    elif measurement == "mem":
        cond, field, results, gb = contextual_fields_analyse(measurement, "", cond, gb,client)


    elif measurement == "ga_page_tracking" :
        cond, field, results, gb = contextual_fields_analyse(measurement, "viewName", cond, gb,client)

    elif measurement == "ga_real_time" :
        cond, field, results, gb = contextual_fields_analyse(measurement, "report", cond, gb,client)


    elif measurement == "accesLogs" :
        cond, field, results, gb = contextual_fields_analyse(measurement, "client", cond, gb,client)


    elif measurement == "ga_sites_users" :
        cond, field, results, gb = contextual_fields_analyse(measurement, "viewName", cond, gb,client)


    elif measurement == "mandrill_email_usage" :
        cond,field,results,gb=contextual_fields_analyse(measurement,"api_key_name",cond,gb,client)

    elif measurement == "healthCheck" :
        cond,field,results,gb=contextual_fields_analyse(measurement,"service",cond,gb,client)

    elif measurement == "appErrors" :
        cond,field,results,gb=contextual_fields_analyse(measurement,"client",cond,gb,client)

    elif measurement == "ga_site_speed" :
        cond,field,results,gb=contextual_fields_analyse(measurement,"viewName",cond,gb,client)

    elif measurement == "net" or  measurement == "http_response" or  measurement == "kernel":
        cond, field, results, gb = contextual_fields_analyse(measurement, "", cond, gb,client)


    else :
        cond=cond +""

    return cond,field,results,gb


def applied_model(Model, df, dfa_2, period, host, measurement, path, form):
    try:
        model=select_apply_model(Model, df, dfa_2, period, host, measurement, path, form)
    except Exception:
        st.write(
            " PROBLEME : Problème lors de l'entrainement du modèle. Le modèle ne convient pas aux données (par exemple SeasonalAD avec des données non cycliques)")
        st.write(
            " PROBLEME : SI vous utilisez le modele IA, vous devez extraire au moins deux semaines de données ")
    return model


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


