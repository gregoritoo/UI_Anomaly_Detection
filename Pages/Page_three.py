import streamlit as st
from Functions.functions_interface import connect_database, get_field_names,select_apply_model
from Functions.functions_requests import make_sliced_request_multicondition, modifie_df_for_fb,load_data_for_analyse
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from Pages.Page import Page


def load_data_for_analyse(host, db, measurement, period, gb, cond, nb_week_to_query, typo, dic, field):
    dfa, client = make_sliced_request_multicondition(host, db, measurement, period, gb, cond, nb_week_to_query,
                                                     typo,
                                                     dic, field)

    df, count, count_2 = modifie_df_for_fb(dfa, typo)
    return df, count, count_2


class Page_three(Page):

    def __init__(self,db):
        Page.__init__(self, db)
        print("Switching to page three")


    def get_data(self,host, db, measurement, period, gb, cond, nb_week_to_query, typo, dic, field):
        df, count ,count_2 = load_data_for_analyse(host, db, measurement, period, gb, cond, nb_week_to_query, typo, dic, field)
        return df,count ,count_2

    def test_correlation(self,df,df_2):
        try:
            plt.scatter(df["y"], df_2["y"])
            st.pyplot()
            st.write("Le coefficient de corrélation des deux variables est : ", pearsonr(df["y"], df_2["y"])[0])
            if abs(pearsonr(df["y"], df_2["y"])[0]) > 0.5:
                st.write('Ces deux données semblent corrélées ')
            elif abs(pearsonr(df["y"], df_2["y"])[0]) < 0.5:
                st.write('Ces deux données ne semblent pas (ou faiblement) corrélées')
        except NameError:
            st.write("Pas de données à étudier")

    def make_second_side_componant(self,db, write_request):

            gb = "host,"
            client = connect_database(db)
            L2 = client.get_list_measurements()
            measurement_2 = st.sidebar.selectbox(
                " Choose measurement  2 ",
                [L2[j]["name"] for j in range(len(L2))])
            results = client.query('SHOW TAG VALUES WITH KEY = "host"')
            P = get_field_names(results, measurement_2)
            host_2 = st.sidebar.selectbox(
                " Choose host 2  ",
                P[:, 1]
            )

            cond = " AND host='" + host_2 + "'"

            cond, field, results, gb = write_request(measurement_2, cond, gb, client,2)

            typo_2 = st.sidebar.selectbox(
                " Choose the aggregation function 2 ",
                ("mean", "max", "sum", "min")
            )

            if measurement_2 == "healthCheck" or measurement_2 == "appErrors":
                typo_2 = field

            dic = ""
            return measurement_2, dic, field, cond, gb, results, typo_2, host_2

    def anomaly_detection(self, form, measurement, df, dfa_2, period, host):
        path=""
        Model = st.sidebar.selectbox(
            " Choose model 3 ",
            ("RegressionResidual", "InterQuartileRangeAD", "PersistAD", "LevelShiftAD", "VolatilityShiftAD",
             "AutoregressionAD")
        )

        model = select_apply_model(Model, df, dfa_2, period, host, measurement, path, form)
        save = st.button("Save model")
        return model