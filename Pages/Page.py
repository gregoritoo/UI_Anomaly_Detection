import streamlit as st
from Functions.functions_interface import connect_database, get_field_names
import pickle
from Alertes.Alert_IA import Alert_IA
from Alertes.Alert import Alert
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import time

class Time():
    def __init__(self, time):
        self.influx_time = time
        if time[-1] == "m":
            self.pandas_time = time[: -1] + "min"
        else:
            self.pandas_time = time


class Page():
    def __init__(self,db):
        print("Switching to page one")

    def make_side_component(self,db,write_request,analyse=False):
        nb_pas = st.number_input('Select frequency (in minutes)')
        group_by_time=int(nb_pas)
        period = str(int(nb_pas)) + "m"

        time_obj = Time(period)

        gb = "host,"
        client = connect_database(db)
        L = client.get_list_measurements()
        measurement = st.sidebar.selectbox(
            " Choose measurement ",
            [L[i]["name"] for i in range(len(L))])
        results = client.query('SHOW TAG VALUES WITH KEY = "host"')
        P = get_field_names(results, measurement)
        host = st.sidebar.selectbox(
            " Choose host ",
            P[:, 1]
        )

        cond = " AND host='" + host + "'"
        cond, field, results, gb = write_request(measurement, cond, gb, client,1,analyse)

        typo = st.sidebar.selectbox(
            " Choose the aggregation function ",
            ("mean", "max", "sum", "min")
        )

        if measurement == "healthCheck" or measurement == "appErrors":
            typo = field

        dic = ""
        nb_week_to_query = st.slider(' nb_week_to_query ')
        return measurement, dic, field, cond, gb, results, typo, period, host, nb_week_to_query, nb_pas,group_by_time

    def plot_pie(self,List,cond):
        plt.pie(x=[List[i]["y"].sum() for i in range(len(cond))],
                autopct='%1.1f%%', labels=[cond[i].split("=")[-1] for i in range(len(cond))], shadow=True)
        plt.title("Répartition de l'activité des éléments choisis")
        plt.show()
        st.pyplot()

    def plot_line_chart(self,List,cond):
        for i in range(len(cond)) :
            rolling_mean=List[i].rolling(window=24).mean()
            plt.plot(List[i]["y"],label=cond[i].split("=")[-1])
            plt.plot(rolling_mean["y"],label="MA de "+cond[i].split("=")[-1])
            plt.legend()
        plt.show()
        st.pyplot()

    def plot_hist(self,List,cond):
        for i in range(len(cond)):
            plt.hist(List[i]["y"], bins=int(int(List[i]["y"].max() - List[i]["y"].min()) / 5) + 1,
                     label=cond[i].split("=")[-1])
            plt.title("Distribution des valeurs")
        plt.legend()
        plt.show()
        st.pyplot()

    def plot_barchart_day(self,List,cond,COUNT):
        jour = ["Dimande", "lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"]
        _X = np.arange(len(jour))
        for i in range(len(cond)):
            plt.bar(_X + 0.2 * i, COUNT[i][0, : 7] / List[i]["y"].sum(), label=cond[i].split("=")[-1], width=0.25)
            plt.xticks(_X, jour)
        plt.legend()
        plt.title("Pourcentage des valeurs en fonction des jours ")
        plt.show()
        st.pyplot()

    def plot_bachart_hours(self,List,cond,COUNT_2):
        heures = [str(i) for i in range(24)]
        _X = np.arange(len(heures))
        for i in range(len(cond)):
            plt.bar(_X + 0.2 * i, COUNT_2[i][0, :] / List[i]["y"].sum(), label=cond[i].split("=")[-1], width=1 / (5))
            plt.xticks(_X, heures)
        plt.legend()
        plt.title("Pourcentage des valeurs en fonction des heures ")
        plt.show()
        st.pyplot()

    def plot_boxchart(self,List,cond):
        nrows = max(int(len(cond)) - 1, 1)
        plt.subplot(nrows, 2, len(cond))
        for i in range(len(cond)):
            ax = plt.subplot(nrows, 2, i + 1)
            ax.boxplot(List[i]["y"])
            plt.title(cond[i].split("=")[-1])
        plt.show()
        st.pyplot()

        for i in range(len(cond)):
            vals = List[i]["y"].describe()
            st.write("RESUME : ")
            st.write("25% des valeurs de " + cond[i].split("=")[-1] + " sont inférieurs à : " + str(vals["25%"]))
            st.write("50% des valeurs de " + cond[i].split("=")[-1] + " sont inférieus à " + str(vals["50%"]))
            st.write("75% des valeurs de " + cond[i].split("=")[-1] + " sont inférieurs à " + str(vals["75%"]))




    def save_or_apply_model(self,model,save,Model,host,measurement,form,field,typo,db,nb_pas,period,gb,cond,c):
        freq_period = st.number_input(
            "Taille du motif qui se répète ; il doit être un nombre impaire qui correspond à la fréquence d'échantillonage")
        force_ml_model = "No"
        force_ml_model = st.selectbox(
            " Forcer modèle de machine learning ? ",
            ("Yes", "No")
        )
        prediction = st.button("Launch ML model training")
        st.write("Ceci va lancer l'entrainement ainsi que la prédiction de la mesure pour les trois prochain mois")

        if save and Model != "Model_VAR_LSTM" :
            file = ""
            if type(form) != list:
                forma = form[1:].split(",")
            else:
                forma = form
            try:
                for element in forma:
                    value = element.split("=")
                    file = file + '_' + value[1]
                file = file[1:].replace(":", "")
            except Exception:
                file = host
            file = file.replace(" ", "")
            path_to_model = r'Modeles_AD/model_' + form[1:].replace(" ","") + '.sav'
            pickle.dump(model, open(path_to_model, 'wb'))
            alerte = Alert(host, measurement)
            message='{{ .Level }} Point anormal associe a host : '+ host + file + "_" + " champs = "+"_"+field+'avec une valeur de {{ index .Fields "value" }} à :{{ .Time }}'
            alerte.create(message, form, nb_pas, path_to_model, field, typo, db)
            alerte.save_alert()
            time.sleep(1)
            alerte.launch()
            st.write("Model saved and alert created !")
            save = False
        elif save and Model == "Model_VAR_LSTM":
            file = ""
            form_before=form
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
            path_to_model = "Modeles_AD/" + file + "_" + measurement
            message = '{{ .Level }} Point anormal associe a host : '+ host + file + "_" + " champs = "+"_"+field+'avec une valeur de {{ index .Fields "value" }} à :{{ .Time }}'
            alerte = Alert_IA(host, measurement)
            with open(path_to_model.split("/")[0] + "/" + path_to_model.split("/")[1] + '/params_var_model.txt',
                      'r') as txt_file:
                look_back = txt_file.read()
            alerte.create( message, form_before, int(period[: -1]), path_to_model, field, typo, db, int(look_back), c)
            alerte.save_alert()
            time.sleep(1)
            alerte.launch()
            st.write("Model saved and alert trigged !")
            save = False
        elif prediction:
            import sys
            b = " " + form + "' '" + period + "/" + host + "/" + measurement + "/" + db + "/" + str(
                int(freq_period)) + "/" + gb + "' '" + field + "' '" + typo + "' '" + cond.replace("'", "/") + "'"
            cmd = b + " \'" + force_ml_model
            print(cmd)
            proc1 = subprocess.Popen(["cmd_IA.bat", cmd], stderr=sys.stderr)  # , creationflags=CREATE_NEW_CONSOLE
            prediction = False


