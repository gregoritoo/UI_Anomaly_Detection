

import streamlit as st
import os
from Pages.Page_two import Page_two
from Pages.Page_one import Page_one
from Functions.functions_interface import  write_request,write_request_analyse
from Functions.functions import transform_time
from Pages.Page_three import Page_three
from Pages.Page_four import Page_four
import plotly.graph_objects as go




path_to_bat = os.environ["Scripts_UI"]


class Time():
    def __init__(self, time):
        self.influx_time = time
        if time[-1] == "m":
            self.pandas_time = time[: -1] + "min"
        else:
            self.pandas_time = time


PAGES = {
    "Anomalie et Prédiction": 1,
    "Data Analyse ": 2,
    "Data Analyse Multivarié": 3,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]
db = "telegraf"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import SpanSelector
from Simple_lstm_new_predictor import Simple_lstm_new_predictor

def onselect(xmin, xmax):
    print("hello")
if page == 1:

        first_page=Page_one(db)
        measurement, dic, field, cond, gb, results, typo, period, host, nb_week_to_query, nb_pas,group_by_time=first_page.make_side_component(db,write_request)
        date_range = st.text_input("select problematique time range", "2019-09-01/2019-09-30")
        form,df,dfa_2=first_page.get_data(host, db, measurement, period, gb, cond, nb_week_to_query, typo, dic, field, date_range)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=list(df.index), y=list(df.y)))
        fig.update_layout(
            title_text="Time series with range slider and selectors"
        )
        st.plotly_chart(fig)
        model,Model,save=first_page.anomaly_detection(form, measurement, df, dfa_2, period, host)

        first_page.save_or_apply_model(model,save,Model,host,measurement,form,field,typo,db,group_by_time,period,gb,cond)
        taille_motif_2 = st.number_input(
            "Taille du motif pour la detection d'anomalie qui se répète ie nombre impaire qui correspond à la fréquence d'échantillonage")
        tat=st.button("Test")
        if tat :

            first_page.create_IA_for_AD(form, host, measurement, db,int(taille_motif_2), period,int(taille_motif_2),gb,cond,typo,field)
elif page == 2:
    try :
        second_page = Page_two(db)
        measurement, dic, field, cond, gb, results, typo, period, host, nb_week_to_query, nb_pas,group_by_time= second_page.make_side_component(
            db,write_request_analyse)
        List,COUNT,COUNT_2=second_page.get_data(host, db, measurement, period, gb, cond, nb_week_to_query, typo, dic, field)
        second_page.plot_pie(List,cond)
        second_page.plot_line_chart(List,cond)
        second_page.plot_hist(List,cond)
        second_page.plot_barchart_day(List,cond,COUNT)
        second_page.plot_bachart_hours(List,cond,COUNT_2)
        second_page.plot_boxchart(List,cond)
    except Exception:
        st.write("La requête ne renvoies rien, merci de vérifier les paramètres")



elif page == 3:
    try :
        third_page=Page_three(db)
        measurement, dic, field, cond, gb, results, typo, period, host, nb_week_to_query, nb_pas ,group_by_time= third_page.make_side_component(
            db, write_request)
        measurement_2, dic, field_2, cond_2, gb_2, results, typo_2, host_2 = third_page.make_second_side_componant( db, write_request)
        df,count,count_2= third_page.get_data(host, db, measurement, period, gb, cond, nb_week_to_query, typo, dic, field)
        df_2,count_21,count_22=third_page.get_data(host_2, db, measurement_2, period, gb_2, cond_2, nb_week_to_query, typo_2, dic, field_2)
        cond=["field="+field,"field="+field_2]
        List=[df,df_2]
        COUNT=[count,count_21]
        COUNT_2=[count_2,count_22]
        third_page.plot_pie(List, cond)
        third_page.plot_line_chart(List, cond)
        third_page.plot_hist(List, cond)
        third_page.plot_barchart_day(List, cond, COUNT)
        third_page.plot_bachart_hours(List, cond, COUNT_2)
        third_page.plot_boxchart(List, cond)
        try :
            third_page.test_correlation(df,df_2)
        except Exception :
            st.write("Probleme avec le calcul de correlation")
    except Exception :
        st.write("La requête ne renvoie rien, merci de vérifier les paramètres")


