import streamlit as st
from Functions.functions_interface import applied_model
import os
from Functions.functions_requests import make_form
from Pages.Page import Page
from Functions.functions_requests import load_data
from Functions.functions import make_sliced_request_multicondition,transform_time,modifie_df_for_fb



class Page_one(Page):
    def __init__(self,db):
        self.db=db
        Page.__init__(self,self.db)

    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def get_data(self,host, db, measurement, period, gb, cond, nb_week_to_query, typo, dic, field,date_range):
        df, dfa_2 =load_data(host, db, measurement, period, gb, cond, nb_week_to_query, typo, dic, field,date_range)
        form = make_form(df, host)
        form = cond.replace(" AND ", ",").replace("\"", "").replace("'", "").replace(":", "")
        return form,df,dfa_2

    def anomaly_detection(self, form, measurement, df, dfa_2, period, host):
        file = ""
        for element in form[1:].split(","):
            value = element.split("=")
            file = file + '_' + value[1]
        file = file[1:].replace(":", "").replace(" ", "")

        path = r"Modeles/" + file + "_" + measurement
        Model = st.sidebar.selectbox(
            " Choose model ",
            ("SeasonalAD", "InterQuartileRangeAD", "PersistAD", "LevelShiftAD", "VolatilityShiftAD",
             "AutoregressionAD","Modele_custom","Model_VAR_LSTM")
        )
        model = applied_model(Model, df, dfa_2, period, host, measurement, path, form)
        save = st.button("Save model")
        return model, Model, save


