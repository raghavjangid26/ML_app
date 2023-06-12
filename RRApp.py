# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import streamlit as st
page_image_bg = f"""
<style>
[data-testid="stAppViewContainer"]{{
background-image: url("https://www.bing.com/th/id/OGC.2cd6ab068da3861d9e16adf23a83ed82?pid=1.7&rurl=https%3a%2f%2fmiro.medium.com%2fmax%2f2800%2f0*hpyUPaBF9V3Mb5T6.gif&ehk=MYrRK08M9fvRKTBulXwxcTs4wy5qVgrQrZRVm7Xr46w%3d");
background-size: cover;
background-repeat: no repeat;
}}

[data-testid="stHeader"]{{
background-image: url("https://www.bing.com/th/id/OGC.2cd6ab068da3861d9e16adf23a83ed82?pid=1.7&rurl=https%3a%2f%2fmiro.medium.com%2fmax%2f2800%2f0*hpyUPaBF9V3Mb5T6.gif&ehk=MYrRK08M9fvRKTBulXwxcTs4wy5qVgrQrZRVm7Xr46w%3d");
background-size: cover;
background-repeat: no repeat;
}}

[data-testid="stSidebar"]{{
background-image: url("https://th.bing.com/th/id/OIP.sXp4Ibf6XcsyZ-0m7xo_MgHaH_?pid=ImgDet&rs=1");
background-size: cover;
background-repeat: no repeat;
}}
</style>
"""
st.markdown(page_image_bg , unsafe_allow_html = True)
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
        st.download_button("Download the file",f,MLmodel.pkl)
        
    
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