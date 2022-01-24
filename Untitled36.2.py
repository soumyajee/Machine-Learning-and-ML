#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import time
df = pd.read_csv("iris.csv")
st.dataframe(df)
st.dataframe(df, 300, 300)
st.dataframe(df.style.highlight_max(axis=0))
st.table(df)
st.write(df.head())
species=st.selectbox("Species:",['setosa','versicolor','virginica'])
st.write("Your Species is:",species)
import streamlit as st
ncol = st.sidebar.number_input("Number of species in column", 0, 3, 1)
cols = st.beta_columns(ncol)
for i, x in enumerate(cols):
    x.selectbox(f"Input # {i}",[1,2,3], key=i)
x = "http://localhost:8501/"
refreshrate =3
refreshrate = int(refreshrate)
while True:
    time.sleep(refreshrate)   


# In[ ]:




