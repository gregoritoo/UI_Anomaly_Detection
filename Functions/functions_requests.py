# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:05:20 2020

@author: GSCA
"""

import pandas as pd

import numpy as np

from Query_3 import Query_all

import os

from datetime import datetime

import streamlit as st

INFLUX_NAME = os.environ['INFLUX_NAME']
INFLUX_PASSWORD = os.environ['INFLUX_PASSWORD']
INFLUX_PORT = 443
INFLUX_HOST = "influxdb-monitoring.supralog.com"
INFLUX_SSL = True
INFLUX_VERIFY_SSL = True

from scipy.stats import anderson

def load_data_for_analyse(host, db, measurement, period, gb, cond, nb_week_to_query, typo, dic, field):
    dfa, client = make_sliced_request_multicondition(host, db, measurement, period, gb, cond, nb_week_to_query,
                                                     typo,
                                                     dic, field)

    df, count, count_2 = modifie_df_for_fb(dfa, typo)
    return df, count, count_2

def load_data(host, db, measurement, period, gb, cond, nb_week_to_query, typo, dic, field,date_range):
    dfa, client = make_sliced_request_multicondition(host, db, measurement, period, gb, cond, nb_week_to_query, typo,
                                                     dic, field)
    dfa_2, client = make_sliced_request_multicondition_range(host, db, measurement, period, gb, cond, nb_week_to_query,
                                                             typo, dic, date_range, field)
    if not isinstance(dfa, pd.DataFrame):
        raise NameError
    df ,_,_= modifie_df_for_fb(dfa, typo)
    dfa_2,_,_ = modifie_df_for_fb(dfa_2, typo)
    return df, dfa_2

def modifie_df_for_fb(dfa, typo):
    count = np.zeros((1, 7))
    count_2 = np.zeros((1, 24))
    if type(dfa) == list:
        df = dfa[0]
        TIME = [None] * len(df['time'])
        # modification du timestamp en format compréhensible par le modèle
        for i in range(len(df['time'])):
            dobj_a = datetime.strptime(df['time'][i], '%Y-%m-%dT%H:%M:%SZ')
            dobj_a.replace(tzinfo=None)
            # dobj=dobj+ timedelta(hours=2)
            dobj = dobj_a.strftime('%Y-%m-%d %H:%M:%S')
            number_day = dobj_a.strftime('%w')
            hour=dobj_a.strftime('%H')
            count[0, int(number_day)] = count[0, int(number_day)] + df[typo][i]
            count_2[0, int(hour)] = count_2[0, int(hour)] + df[typo][i]
            TIME[i] = dobj

        for i in range(len(dfa)):
            dfa[i]["time"] = TIME
            dfa[i] = dfa[i].rename(columns={"time": "ds", typo: "y"})
    else:
        df = dfa
        TIME = [None] * len(df['time'])
        # modification du timestamp en format compréhensible par le modèle
        for i in range(len(df['time'])):
            dobj_a = datetime.strptime(df['time'][i], '%Y-%m-%dT%H:%M:%SZ')
            dobj_a.replace(tzinfo=None)
            # dobj=dobj+ timedelta(hours=2)
            dobj = dobj_a.strftime('%Y-%m-%d %H:%M:%S')
            number_day = dobj_a.strftime('%w')
            hour = dobj_a.strftime('%H')
            count[0, int(number_day)] = count[0, int(number_day)] + df[typo][i]
            count_2[0, int(hour)] = count_2[0, int(hour)] + df[typo][i]
            TIME[i] = dobj

        df["time"] = TIME
        df = df.reset_index(drop=True)
        dfa = df.rename(columns={"time": "ds", typo: "y"})
    return dfa,count,count_2






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
        result, client = query.request(every, period, cond, gb, dic)
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
<<<<<<< HEAD
        if len(gb) < 6:
            for k in range(len(week) - 1):
                every = " now() -" + str(int(week[k + 1])) + 'w AND "time" < now() - ' + str(int(week[k])) + 'w '
                result, client = query.request(every, period, cond, gb, dic)
                df = result[host]
                li[len(week) - 1 - k] = df
            df = pd.concat(li, axis=0, join="outer")
            df = df.reset_index()
            lli = df[["time", typo]]
            print(lli)
        else:
            for k in range(len(week) - 1):
                every = " now() -" + str(int(week[k + 1])) + 'w AND "time" < now() - ' + str(int(week[k])) + 'w '
                result, client = query.request(every, period, cond, gb, dic)
                li[len(week) - 1 - k] = result
            if result != -1:
                dfs = [[None] * (len(week) - 1)] * len(result)
                t = 0
                lli = [None] * len(result)
                if len(result) > 1:
                    for i in range(len(result)):
                        for j in range(1, len(week)):
                            dfs[t][j - 1] = li[j][i]
                        lli[i] = pd.concat(dfs[0], axis=0, join="outer").reset_index()
                        t = t + 1
                        print(lli)
                else:
                    for j in range(1, len(week)):
                        li[j] = pd.DataFrame(result[0])
                    lli = pd.concat(li, axis=0, join="outer").reset_index()
                    lli = lli[["time", typo]]
                    print(lli)
            else:
                lli = None
=======
        for k in range(len(week) - 1):
            every = " now() -" + str(int(week[k + 1])) + 'w AND "time" < now() - ' + str(int(week[k])) + 'w '
            df, client = query.request(every, period, cond, gb, dic)
            li[len(week) - 1 - k] = df
        df = pd.concat(li, axis=0, join="outer")
        df = df.reset_index()
        lli = df[["time", typo]]
>>>>>>> old-version
        return lli, client
    except UnboundLocalError:
        if UnboundLocalError:
            st.write("Choisir l'ensemble des paramètres")


def make_sliced_request_multicondition_range(host, db, measurement, period, gb, cond, past, typo, dic, date_range,
                                             field):
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
    query = Query_all(db, host, typo, field, measurement)
    week = np.linspace(0, past, past + 1)
    range = date_range.split("/")
    li = [None] * len(week)
    every = "'" + str(range[0]) + "'" + ' AND "time" < ' + "'" + str(range[1]) + "'"
    df, client = query.request(every, period, cond, gb, dic)
    df = df.reset_index()
    lli = df[["time", typo]]
    return lli, client



def make_form(df, host):
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
    if type(df) == list:
        df = df[0]
    columns = df.columns
    df = df.reset_index()
    form = ""
    for j in range(len(columns)):  # utilise les colonnes pour pouvoir adapter l'ecriture dans la table
        if columns[j] != "ds" and columns[j] != "mean" and columns[j] != "index" and columns[j] != "y":
            form = form + "," + columns[j] + "=" + str(df[columns[j]][0])
    print(form)
    if len(form) < 1:
        form = ",host=" + host
    return form


def transform_time(period):
    if period[-1] == "s":
        nb_points = int(period[0:len(period) - 1])
    elif period[-1] == "m":
        nb_points = int(60 / int(period[0:len(period) - 1])) * 24 * 7
    elif period[-1] == "h":
        nb_points = int(period[0:len(period) - 1]) * 24 * 7
    elif period[-1] == "d":
        nb_points = int(period[0:len(period) - 1]) * 7
    elif period[-1] == "w":
        nb_points = int(period[0:len(period) - 1]) * 1
    return (nb_points)

