import streamlit as st
import pandas as pd  #para manejar dataframes



def tables():
    st.title("TABLES")
    
    file=st.file_uploader("load csv",type=["csv"]) #carga de archivos
    if file is not None:   #si no existe el archivo 
        csv= pd.read_csv(file)  #lee el archivo
        st.subheader("Cars")
              
        # # solo limitar la vista por slider y mostrar
        # limit=st.slider("limit rows:",min_value=10,max_value=len(csv),value=10 ) #value final es el valor por cual empieza la prox
        # tables_limit=csv.head(limit)  #limita la vista del dataframe
        # st.write(f"number rows:{limit}")  #muestra el numero de filas 
        # st.table(tables_limit)  #muestra la tabla
        
        
        #eliminar columnas o limitar vista y mostrarb tabla LIMITADA
        drop_columns=st.multiselect("drop columns:",options=csv.columns )  #tenemos disponibles las columnas
        if drop_columns:  #si hay columnas para eliminar
            #limitar la vista por slider
            limit=st.slider("limit rows:",min_value=10,max_value=len(csv),value=10 ) #value final es el valor por cual empieza la prox
            tables_limit=csv.head(limit)  #limita la vista del dataframe
            st.write(f"number rows:{limit}")  #muestra el numero de filas
            drop=tables_limit.drop(columns=drop_columns)  #elimina las columnas seleccionadas   
            st.table(drop)  #muestra la tabla
            #INICIALMENTE YA 
        else:  #si no hay columnas para eliminar
            hide_columns=["id","color"]
            csv_final=csv.drop(columns=hide_columns,errors="ignore")  #elimina las columnas seleccionadas
            st.table(csv_final.head(10))  #muestra la tabla
  
        
    else:
        st.info("Load a csv file")
            
            
    
