import streamlit as st
from Pages.Page import Page
from Functions.functions_requests import load_data_for_analyse






class Page_four(Page):
    def __init__(self,db):
        Page.__init__(self,db)
        print("Switching to page two")


    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def get_data(self,host, db, measurement, period, gb, cond, nb_week_to_query, typo, dic, field):
        List = [None] * len(cond)
        COUNT = [None] * len(cond)
        COUNT_2 = [None] * len(cond)
        i = 0

        for condition in cond:
            df, count, count_2 = load_data_for_analyse(host, db, measurement, period, gb, condition,
                                                       nb_week_to_query, typo, dic, field)
            List[i] = df
            COUNT[i] = count
            COUNT_2[i] = count_2
            i = i + 1
        return List,COUNT,COUNT_2

    def get_all_data(self,host, db, measurement, period, gb, cond, nb_week_to_query, typo, dic, field,L,M):
        List = [None] * len(L)
        i=0
        for field in L:
            gb = "host," + M[i] + ","
            df,_ , _, _ = load_data_for_analyse(host, db, M[i], period, gb, cond,
                                                       nb_week_to_query, typo, dic, field)
            List[i] = df
            i=i+1
        return List