import streamlit as st
#importamos archivos 
from tables import tables as t
from charts import charts as c
# t()  #probamos que funciona
# c()  #probamos que funciona

#otra forma de importar archivos
import dataframe as df
# df.dataframes()  #probamos que funciona  

#determinamos el menu
if "menu" not in st.session_state:
    st.session_state.menu="TABLES"  #valor por defecto

#sirebar
with st.sidebar:
    st.session_state.menu=st.selectbox("GIBIO FRBA",["DATAFRAMES","TABLES","CHARTS"])
    
if st.session_state.menu=="TABLES":
    t()
elif st.session_state.menu=="CHARTS":
    c()
elif st.session_state.menu=="DATAFRAMES":
    df.dataframes()
     
    




    

