# import streamlit as st
# import pandas as pd

# # # solo me muestra solo 10 datos sin limitArlo, es dinamica y puedo ordenar las columnas
# # def dataframes():
# #     st.title("DATAFRAMES")
    
# #     file=st.file_uploader("load csv",type=["csv"]) #carga de archivos
    
# #     if file is not None:   #si no existe el archivo
# #         csv=pd.read_csv(file)  #lee el archivo
# #         st.subheader("Cars")
# #         st.dataframe(csv)  #muestra el archivo completo
# #     else:
# #         st.info("Load a csv file")



# # #pongo un buscador para filtrar datos ADICIONAL 
# # def dataframes():
# #     st.title("DATAFRAMES")
# #     file=st.file_uploader("load csv",type=["csv"]) #carga de archivos
    
# #     if file is not None:   #si no existe el archivo
# #         csv=pd.read_csv(file)  #lee el archivo
        
# #         search=st.text_input("", placeholder="search...", autocomplete="off") #me devuelve true o false
# #         if search:#filtro
# #             #apply es de panda (aplica una funcion directamente a mis datos), lambda es funcion anonima, row es la pila de cada registro
# #             #astype=convertir todos los registros en STRINGS,
# #             #aplicamos otra funcion str.contains()=para que contenga todo  el strings esta lo que busco(search), case=false es para que no tome mayusculas
# #             #any=nos devuelve un true o false en cada fila que buscamos
# #             #axis=1, es para que se ejecute en cada una de las filas que tengo el archivo
# #             filter_data=csv[csv.apply(lambda row: row.astype(str).str.contains(search, case=False).any(),axis=1)]   
# #         else:
# #             filter_data=csv #si no hay texto de busqueda, muestra el archivo original csv
        
# #         st.subheader("Cars")
# #         st.dataframe(filter_data)  #muestra el dato  buscado 
# #     else:
# #         st.info("Load a csv file")
        

      
# #pongo un buscador para filtrar datos ESPECIFICOS, ya que al buscar RED , busca la palabra y me suma "tredia" en modelo erroneamente
# def dataframes():
#     st.title("DATAFRAMES")
#     file=st.file_uploader("load csv",type=["csv"]) #carga de archivos
    
#     if file is not None:   #si no existe el archivo
#         csv=pd.read_csv(file)  #lee el archivo
#         field=st.selectbox("", options=csv.columns) #selecciono la columna donde quiero buscar
#         search=st.text_input("", placeholder="search...", autocomplete="off") #me devuelve true o false
        
#         if search:#filtro
#             #search.lower()=convierte todo a minusculas
#             filter_dataEspecif=csv[csv.apply(lambda row:search.lower())]
#         else:
#             filter_data=csv #si no hay texto de busqueda, muestra el archivo original csv
            


    
    
  
         
    
    
    
    
    
    
    
    
    
    
    
    
    
  
        