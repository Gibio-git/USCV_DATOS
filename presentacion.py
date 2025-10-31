# app.py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Histos dinámicos", layout="wide")

st.title("Histogramas dinámicos (EDAD / IMC)")

# --- Carga de datos ---
st.sidebar.header("Datos")
archivo = st.sidebar.file_uploader("Subí tu CSV", type=["csv"])
sep = st.sidebar.text_input("Separador (ej: , ; \\t)", value=",")
decimal = st.sidebar.text_input("Separador decimal (., ,)", value=".")

@st.cache_data
def _leer_csv_(file, sep, decimal):
    return pd.read_csv(file, sep=sep, decimal=decimal)

if archivo is not None:
    df = _leer_csv_(archivo, sep, decimal)
else:
    st.info("No subiste CSV. Uso un ejemplo sintético con columnas: EDAD, IMC, SEXO.")
    np.random.seed(7)
    df = pd.DataFrame({
        "EDAD": np.random.randint(18, 80, 400),
        "IMC": np.round(np.random.normal(26, 4, 400).clip(15, 40), 1),
        "SEXO": np.random.choice(["F", "M"], 400, p=[0.55, 0.45])
    })

# Intentar mapear nombres habituales
posibles_cols = {
    "EDAD": [c for c in df.columns if c.strip().upper() in ["EDAD","EDAD (AÑOS)","AGE"]],
    "IMC":  [c for c in df.columns if c.strip().upper() in ["IMC","BMI"]],
    "SEXO": [c for c in df.columns if c.strip().upper() in ["SEXO","SEX","GENERO","GÉNERO"]]
}
EDAD_col = posibles_cols["EDAD"][0] if posibles_cols["EDAD"] else None
IMC_col  = posibles_cols["IMC"][0]  if posibles_cols["IMC"]  else None
SEXO_col = posibles_cols["SEXO"][0] if posibles_cols["SEXO"] else None

col_map_help = st.sidebar.expander("Mapear columnas (si tus nombres difieren)")
with col_map_help:
    EDAD_col = st.selectbox("Columna EDAD", [None]+list(df.columns), index=(list(df.columns).index(EDAD_col)+1) if EDAD_col in df.columns else 0)
    IMC_col  = st.selectbox("Columna IMC",  [None]+list(df.columns), index=(list(df.columns).index(IMC_col)+1)  if IMC_col in df.columns  else 0)
    SEXO_col = st.selectbox("Columna SEXO", [None]+list(df.columns), index=(list(df.columns).index(SEXO_col)+1) if SEXO_col in df.columns else 0)

# Detectar numéricas y categóricas
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if pd.api.types.is_categorical_dtype(df[c]) or df[c].dtype == "object"]

# --- Controles de gráfico ---
st.sidebar.header("Gráfico")
variable = st.sidebar.selectbox(
    "Variable numérica",
    [c for c in [EDAD_col, IMC_col] if c is not None] or num_cols
)

color_col = st.sidebar.selectbox(
    "Color (hue)", [None] + ([SEXO_col] if SEXO_col else []) + [c for c in cat_cols if c != SEXO_col]
)

# Filtro por SEXO si existe
if SEXO_col:
    valores_sexo = sorted(df[SEXO_col].dropna().unique().tolist())
    seleccion_sexo = st.sidebar.multiselect(f"Filtrar {SEXO_col}", valores_sexo, default=valores_sexo)
    df_filtrado = df[df[SEXO_col].isin(seleccion_sexo)]
else:
    df_filtrado = df.copy()

# Rango y bins
x_min = float(np.nanmin(df_filtrado[variable])) if variable else 0.0
x_max = float(np.nanmax(df_filtrado[variable])) if variable else 1.0

st.sidebar.subheader("Bineado")
modo_bin = st.sidebar.radio("Elegir por…", ["Cantidad de bins", "Tamaño de bin"], index=0)
if modo_bin == "Cantidad de bins":
    nbins = st.sidebar.slider("Cantidad de bins", min_value=3, max_value=100, value=20)
    xbins = None
else:
    tam_def = 1.0 if "IMC" not in (variable or "").upper() else 0.5
    bin_size = st.sidebar.number_input("Tamaño de bin", min_value=0.1, max_value=10.0, value=float(tam_def), step=0.1, format="%.2f")
    nbins = None
    xbins = dict(size=bin_size)

rango = st.sidebar.slider("Rango X", min_value=float(np.floor(x_min)), max_value=float(np.ceil(x_max)),
                          value=(float(np.floor(x_min)), float(np.ceil(x_max))), step=1.0)

modo_barras = st.sidebar.radio("Modo de barras", ["Apareadas (group)", "Superpuestas (overlay)", "Apiladas (stack)"], index=0)
modo_map = {
    "Apareadas (group)": "group",
    "Superpuestas (overlay)": "overlay",
    "Apiladas (stack)": "relative"
}

mostrar_texto = st.sidebar.checkbox("Mostrar etiquetas de conteo", value=False)
opacidad = st.sidebar.slider("Opacidad", 0.1, 1.0, 0.85, 0.05)

# --- Figura ---
if variable is None:
    st.error("Seleccioná una columna numérica para graficar.")
else:
    fig = px.histogram(
        df_filtrado,
        x=variable,
        color=color_col,
        nbins=nbins,
        opacity=opacidad
    )
    # aplicar tamaño de bin si corresponde
    if xbins is not None:
        for tr in fig.data:
            tr.update(xbins=xbins)

    fig.update_layout(
        barmode=modo_map[modo_barras],
        bargap=0.05,
        xaxis_title=variable,
        yaxis_title="Frecuencia",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_xaxes(range=list(rango), tickangle=45)

    if mostrar_texto:
        fig.update_traces(texttemplate="%{y}", textposition="outside", cliponaxis=False)

    st.plotly_chart(fig, use_container_width=True)

# --- Extras útiles ---
with st.expander("Ver primeras filas"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Notas"):
    st.markdown("""
- Si tus columnas no se llaman exactamente **EDAD**, **IMC**, **SEXO**, usá *Mapear columnas* en la barra lateral.
- **Apareadas (group)** ≈ *dodge* de seaborn; **Apiladas (stack)** suma las categorías; **Superpuestas (overlay)** superpone con opacidad.
- Activá *Mostrar etiquetas de conteo* para ver valores encima de las barras.
""")
