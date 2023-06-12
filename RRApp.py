# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib as plt
import seaborn as sns
import os
import pandas_profiling 
from streamlit_pandas_profiling import st_profile_report

from pycaret.regression import pull, save_model, setup, compare_models, RegressionExperiment
from pycaret.classification import pull, save_model, setup, compare_models, ClassificationExperiment


with st.sidebar:
    st.title("Raghav's Regression app")
    choice = st.selectbox("choose:",["Upload","EDA","Model","Download"])

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice=="Upload":
    st.title("Upload your data here")
    dataset_values = st.file_uploader("Upload your dataset")
    if dataset_values:
        df = pd.read_csv(dataset_values, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

if choice=="EDA":
    st.title("Explore Your Data using EDA")
    profile_report = df.profile_report()
    st_profile_report(profile_report)
    

if choice=="Download":
    with open("MLmodel.pkl",'rb') as f:
        st. title("Download your model from here:")
        st.download_button("Download the file",f,"MLmodel.pkl")
        
    
if choice=="Model":
    st.title("Machine Learning for Regrssion Analysis")
    choice = st.sidebar.selectbox("Choose your method",["Classification","Regression"])
    if choice == "Classification":
        target= st.selectbox("select your target", df.columns)
        if st.button("Train"):
            s1 = ClassificationExperiment()
            s1.setup(df, target=target)
            setup_df = s1.pull()
            st.info("This is ML Experiment")
            st.dataframe(setup_df)
            best_model = s1.compare_models()
            compare_df = s1.pull()
            st.info("This is ML model")
            st.dataframe(compare_df)
            best_model
            s1.save_model(best_model, 'Machine Learning Model')
    else:
        if choice == "Regression":
            target= st.selectbox("select your target", df.columns)
            if st.button("Train"):
                s2 = RegressionExperiment()
                s2.setup(df, target=target)
                setup_df = s2.pull()
                st.info("This is ML Experiment")
                st.dataframe(setup_df)
                best_model = s2.compare_models()
                compare_df = s2.pull()
                st.info("This is ML model")
                st.dataframe(compare_df)
                best_model
                s2.save_model(best_model, 'Machine Learning Model')
