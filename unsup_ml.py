# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:40:53 2023

@author: varulobo
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_white"

st.header("Unsupervised Machine Learning Classification using KMeans ")

p="This webpage was created using streamlit library and hosted on heroku.\n \
We use the 'make_blobs' library in scikit-learn to generate random datapoints.\n\
Using the slider bar below you can adjust the sample size and spread."
       
st.markdown(p)

st.subheader("Enter the total sample size")
sample=st.slider("", 1,1000,500, key="sample")

st.subheader("Enter the std of each cluster")
std=st.slider("", 0.1,5.0,0.5, key='std')

st.subheader("Enter the total number of clusters")
centerA=st.slider("", 1,10,4, key='centerA')


from sklearn.datasets import make_blobs
x,y=make_blobs(n_samples=sample, centers=centerA, cluster_std=std, random_state=55)


figA=px.scatter(x=x[:,0],y=x[:,1])
st.plotly_chart(figA)

st.subheader("KMeans Implementation:")

code="\
from sklearn.cluster import KMeans \n\
kmeans=KMeans(n_clusters=4, max_iter=1000) \n\
kmeans.fit(x) \n\
labels=kmeans.predict(x) "

with st.expander("See code:"):
    st.code(code ,language=('python'))

st.subheader("Select total number of clusters to be predicted using KMeans")
centerB=st.slider("", 1,10,4, key='centerB')

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=centerB, max_iter=1000)
kmeans.fit(x)
labels=kmeans.predict(x)


figB=px.scatter(x=x[:,0],y=x[:,1],color=labels,render_mode="svg")
st.plotly_chart(figB)


msg="We have successfully demonstrated the implementation of KMeans unsupervised ML algorithm\
    to classify randomly scattered datapoints"

st.success(msg)





