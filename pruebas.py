# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import unicodedata
from pathlib import Path
import requests
from io import BytesIO, StringIO

st.set_page_config(page_title="GIBIO - Dashboard", layout="wide")

# ===== Configura tu origen de datos =====
RUTA_LOCAL = "baseGeneral.csv"    # poné tu archivo aquí
URL_BACKUP = None                 # opcional: URL RAW si no hay archivo local
SEPARADOR = ";"                   # cambia a "," o "\t" si corresponde
CODIFICACION = "ISO-8859-1"       # o "utf-8"
SKIPROWS = 1                      # 0 si tu CSV no tiene fila extra

# ===== Utilidades =====
def _normalize(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode()
    s = s.strip().upper().replace("-", " ").replace("_", " ")
    while "  " in s:
        s = s.replace("  ", " ")
    return s

def _resolver_columna(df: pd.DataFrame, candidatos):
    norm = {_normalize(c): c for c in df.columns}
    for cand in candidatos:                 # match exacto
        key = _normalize(cand)
        if key in norm:
            return norm[key]
    # match por palabras
    keys = [_normalize(c) for c in candidatos]
    for k, original in norm.items():
        if any(w in k.split() for w in keys):
            return original
    return None

@st.cache_data(ttl=600)
def cargar_datos() -> tuple[pd.DataFrame, str, str]:
    # 1) intenta local
    if Path(RUTA_LOCAL).exists():
        df = pd.read_csv(RUTA_LOCAL, sep=SEPARADOR, encoding=CODIFICACION,
                         skiprows=SKIPROWS, engine="python", on_bad_lines="skip")
    # 2) intenta URL backup (si se definió)
    elif URL_BACKUP:
        r = requests.get(URL_BACKUP, timeout=60)
        r.raise_for_status()
        try:
            df = pd.read_csv(StringIO(r.text), sep=SEPARADOR,
                             encoding=CODIFICACION, skiprows=SKIPROWS,
                             engine="python", on_bad_lines="skip")
        except Exception:
            df = pd.read_excel(BytesIO(r.content))
    else:
        raise FileNotFoundError(f"No encontré {RUTA_LOCAL} y no hay URL_BACKUP.")

    col_sexo = _resolver_columna(df, ["SEXO", "GENERO", "GÉNERO"])
    col_act  = _resolver_columna(df, ["ACTIVIDAD DEPORTIVA", "DEPORTE",
                                      "ACTIVIDAD", "TIPO DE DEPORTE"])

    if not col_sexo or not col_act:
        raise KeyError(
            "No encontré columnas de SEXO/GÉNERO y ACTIVIDAD/DEPORTE.\n"
            f"Disponibles: {list(df.columns)}"
        )

    df[col_sexo] = df[col_sexo].astype(str).str.strip()
    df[col_act]  = df[col_act].astype(str).str.strip()

    return df, col_sexo, col_act

# ===== APP =====
st.title("🏃 Dashboard — Sexo × Actividad Deportiva")

try:
    df, COL_SEXO, COL_ACT = cargar_datos()
except Exception as e:
    st.error(f"Error al cargar datos: {e}")
    st.stop()

# ---- Filtros ----
with st.sidebar:
    st.header("Filtros")
    sexos = sorted([s for s in df[COL_SEXO].dropna().unique().tolist() if s != ""])
    acts  = sorted([a for a in df[COL_ACT].dropna().unique().tolist() if a != ""])
    f_sexo = st.multiselect("Sexo / Género", sexos, default=sexos)
    f_act  = st.multiselect("Actividad deportiva", acts, default=acts)

df_f = df.copy()
if f_sexo:
    df_f = df_f[df_f[COL_SEXO].isin(f_sexo)]
if f_act:
    df_f = df_f[df_f[COL_ACT].isin(f_act)]

# ---- KPIs ----
c1, c2, c3 = st.columns(3)
c1.metric("Registros", f"{len(df_f):,}".replace(",", "."))
c2.metric("Géneros", df_f[COL_SEXO].nunique())
c3.metric("Actividades", df_f[COL_ACT].nunique())

st.markdown("---")

# ---- Gráfico 1: Conteo por actividad segmentado por sexo ----
g1 = (
    df_f.groupby([COL_ACT, COL_SEXO]).size()
        .reset_index(name="Cantidad")
        .sort_values("Cantidad", ascending=False)
)
fig1 = px.bar(
    g1, x=COL_ACT, y="Cantidad", color=COL_SEXO, barmode="group",
    title="Cantidad por actividad (segmentado por sexo)"
)
fig1.update_layout(xaxis_title="ACTIVIDAD DEPORTIVA", yaxis_title="Cantidad",
                   legend_title="SEXO")
st.plotly_chart(fig1, use_container_width=True)

# ---- Gráfico 2: Heatmap Actividad × Sexo ----
pv = pd.pivot_table(df_f, index=COL_ACT, columns=COL_SEXO,
                    values=COL_ACT, aggfunc="count", fill_value=0)
fig2 = px.imshow(pv, text_auto=True, aspect="auto",
                 title="Heatmap — Registros por Actividad × Sexo")
st.plotly_chart(fig2, use_container_width=True)

# ---- Gráfico 3: % dentro de cada sexo (stacked 100%) ----
pct = (
    df_f.groupby([COL_SEXO, COL_ACT]).size()
        .groupby(level=0).apply(lambda s: 100 * s / s.sum())
        .reset_index(name="Porcentaje")
)
fig3 = px.bar(
    pct, x=COL_ACT, y="Porcentaje", color=COL_SEXO, barmode="relative",
    title="Participación % por actividad dentro de cada sexo"
)
fig3.update_layout(xaxis_title="Actividad deportiva", yaxis_title="%", legend_title="Sexo")
st.plotly_chart(fig3, use_container_width=True)

# ---- Tabla resumen + descarga ----
st.subheader("Top actividades (según filtros)")
top = df_f[COL_ACT].value_counts().reset_index()
top.columns = [COL_ACT, "Cantidad"]
st.dataframe(top, use_container_width=True)

st.download_button(
    "⬇️ Descargar datos filtrados (CSV)",
    data=df_f.to_csv(index=False).encode("utf-8"),
    file_name="datos_filtrados.csv",
    mime="text/csv"
)
