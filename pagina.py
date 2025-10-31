import streamlit as st #framework web
import pandas as pd #manejo de dataframes
import matplotlib.pyplot as plt #graficos
import unicodedata, re, numpy as np #normalizacion de textos y operaciones numericas

st.set_page_config(
    page_title="Protocolo GIBIO Dashboard",
    layout="wide",   # <- usa todo el ancho
    initial_sidebar_state="collapsed"  # <- oculta la barra lateral
)

# GENERAL

def clean_headers(cols): #normaliza y limpia encabezados
    return [
        re.sub(r"\s+", " ",
               unicodedata.normalize("NFD", c).encode("ascii", "ignore").decode("utf-8") #saca espacio, tildes, mayusculas
        ).strip().upper() #quita espacios y pasa a mayusculas
        for c in cols #hace una lista con todos los encabezados en mayusculas
    ]

def pick_col(df, rules): #busca columna segun reglas(tokens), df=dataframe, rules=lista de listas de tokens
    """
    Devuelve el nombre de la primera columna cuyo encabezado (upper)
    contenga TODOS los tokens de una de las reglas.
    'rules' es una lista de listas de tokens. Cada sublista es una alternativa.
    ¿Por qué hay dos bucles?
    Externo (for r in rules): recorre las alternativas en orden de preferencia.
    Interno (for c in df.columns): busca una columna que cumpla la alternativa r.
    """
    for r in rules:  #Cada alternativa r es una lista de tokens que deben aparecer en el nombre de la columna, donde rules es una lista de listas de alternativas 
        for c in df.columns: #recorre todas las columnas del dataframe,
            name = c.upper() #convierte el nombre de la columna a mayusculas
            if all(tok in name for tok in r): #si todos los tokens de la regla r estan en el nombre de la columna.Si r = ["VEL", "ONDA", "PULSO"], 
                                              #es True cuando las tres palabras aparecen en el nombre, en cualquier orden (por ejemplo: “Vel. de Onda de Pulso”).
                return c #devuelve el nombre de la columna que cumple con la regla
    return None

def to_number(series: pd.Series) -> pd.Series:
    """Convierte textos como '12,3', '120 mmHg', '10 m/s' a float. Si no hay número, deja NaN."""
    s = series.astype(str).str.replace(",", ".", regex=False) #reemplaza comas por puntos
    s = s.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)  #intenta extraer el primer número que encuentre dentro del texto.
    return pd.to_numeric(s, errors="coerce") #convierte la serie extraida a numerica, si no puede lo convierte en NaN


# boxplot por sexo (genero)

def boxplot_por_sexo(df, col_num, col_sexo, titulo, unidad=""):
    """Devuelve una figura con boxplots por categoría (sexo)."""
    s = pd.to_numeric(df[col_num], errors="coerce")
    cat = df[col_sexo].astype(str)
    data = pd.DataFrame({col_num: s, col_sexo: cat}).dropna(subset=[col_num, col_sexo])

    grupos, etiquetas = [], []
    for g, sub in data.groupby(col_sexo):
        vals = sub[col_num].dropna().values
        if len(vals) >= 2:
            grupos.append(vals)
            etiquetas.append(f"{g} (n={len(vals)})")

    fig, ax = plt.subplots(figsize=(7, 5))
    if len(grupos) == 0:
        ax.text(0.5, 0.5, "Sin datos suficientes para boxplot",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    ax.boxplot(grupos, vert=True, patch_artist=True, labels=etiquetas)
    # puntos individuales con jitter leve
    for i, vals in enumerate(grupos, start=1):
        x = np.random.normal(i, 0.05, size=len(vals))
        ax.plot(x, vals, "o", alpha=0.35, markersize=4)

    ax.set_title(titulo)
    ax.set_ylabel(f"{col_num}" + (f" ({unidad})" if unidad else ""))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig

# =======================
# Carga de datos
# =======================
df = pd.read_excel("EVALUACIONES2024.xlsx", header=1)  # encabezados reales en fila 2 (0-index)
df.columns = clean_headers(df.columns)
# st.write("Columnas detectadas:", list(df.columns))   #SOLO PARA VER LOS DATOS QUE TENGO EN MI INDICE, PRUEBA DEBUG

# =======================
# Detectar columnas clave (nombres flexibles)
# =======================
COL_ACT = pick_col(df, [["ACTIVIDAD","DEPOR"], ["DEPORTE"], ["ACTIVIDAD"]])
COL_SEXO = pick_col(df, [["SEXO"], ["GENERO"]])

# FC: contemplamos "FC 1", "FC"
COL_FC   = pick_col(df, [["FREC","CARDI"], ["FC 1"], ["FC"]])

# PAS y PAD: cada alternativa por separado (no mezclar tokens de distintas opciones)
COL_PAS  = pick_col(df, [["PAS 1"], ["PAS"], ["SISTO"]])
COL_PAD  = pick_col(df, [["PAD 1"], ["PAD"], ["DIASTO"]])

COL_CINT = pick_col(df, [["CINTUR"]])

# VOP: varias alternativas (si tenés VOP 1 / VOP 2 ver bloque opcional más abajo)
COL_VOP  = pick_col(df, [["VOP"], ["VEL","ONDA","PULSO"], ["VEL","ONDA"], ["VEL","PULSO"]])

COL_PCEN = pick_col(df, [["PRESION","CENTRAL"], ["PCENTRAL"], ["PRES CENT"]])
COL_ANT  = pick_col(df, [["ANTECEDENTES","CV"], ["ANTECEDENTES CV"], ["ANTECEDENTES"]])
COL_PESO = pick_col(df, [["PESO"]])
COL_ALT  = pick_col(df, [["ALTURA"]])
COL_IMC  = pick_col(df, [["IMC"]])
# Patología / preexistencia (si existiera)
COL_PATO = pick_col(df, [["PATOLOG"], ["PREEXIST"], ["ENFERM"]])

st.title("📊 Protocolo de Evaluaciones Cardiovasculares — GIBIO FRBA")

if not COL_ACT or not COL_SEXO:
    st.error(f"No encontré columnas de ACTIVIDAD/DEPORTE o SEXO/GÉNERO.\nEncabezados: {list(df.columns)}")
    st.stop()

# # ============ Debug opcional ============
# st.caption(f"Cols detectadas → PAS:{COL_PAS} | PAD:{COL_PAD} | FC:{COL_FC} | VOP:{COL_VOP} | IMC:{COL_IMC}")

# =======================
# Sidebar filtros
# =======================
with st.sidebar:
    st.header("Filtros")
    actividades = sorted([a for a in df[COL_ACT].dropna().astype(str).str.strip().unique() if a!=""])
    sexos       = sorted([s for s in df[COL_SEXO].dropna().astype(str).str.strip().unique() if s!=""])
    f_sexo = st.multiselect("Sexo / Género", options=sexos, default=sexos)
    f_act  = st.multiselect("Actividad / Deporte", options=actividades, default=actividades)
    q = st.text_input("Buscar actividad (contiene)", "")
    mostrar_hist = st.checkbox("Mostrar histogramas de PAS / PAD / VOP", value=True)

# Aplicar filtros
df_f = df.copy()
if f_sexo: df_f = df_f[df_f[COL_SEXO].isin(f_sexo)]
if f_act:  df_f = df_f[df_f[COL_ACT].isin(f_act)]
if q.strip():
    df_f = df_f[df_f[COL_ACT].str.contains(q.strip(), case=False, regex=False)]

# =======================
# Normalización numérica (clave para ver gráficos)
# =======================
NUM_CANDIDATAS = [COL_PAS, COL_PAD, COL_VOP, COL_PCEN, COL_FC, COL_CINT, COL_IMC, COL_PESO, COL_ALT]
for col in NUM_CANDIDATAS:
    if col and col in df.columns:
        df[col] = to_number(df[col])
    if col and col in df_f.columns:
        df_f[col] = to_number(df_f[col])

# (Opcional) Si tenés VOP en dos columnas, descomentá este bloque y ajustá los nombres:
# COL_VOP1 = pick_col(df, [["VOP 1"], ["VOP"]])
# COL_VOP2 = pick_col(df, [["VOP 2"]])
# if COL_VOP1 and COL_VOP2:
#     df_f["VOP_MIX"] = to_number(df_f[COL_VOP1]).fillna(to_number(df_f[COL_VOP2]))
#     COL_VOP = "VOP_MIX"

# =======================
# KPIs
# =======================
# KPIs
# ===========================
c1, c2, c3, c4 = st.columns(4)

# Total de registros
c1.metric("Registros", f"{len(df_f):,}".replace(",", "."))

# Conteo por género (usa 'M' y 'F')
if COL_SEXO in df_f.columns:
    sex_counts = df_f[COL_SEXO].astype(str).str.upper().value_counts()
    fem = sex_counts.get("F", 0)
    masc = sex_counts.get("M", 0)
    otros = sex_counts.sum() - fem - masc
    c2.metric("♀ Femenino (F)", fem)
    c3.metric("♂ Masculino (M)", masc)
    if otros > 0:
        st.info(f"Otros / sin datos: {otros}")
else:
    c2.metric("♀ Femenino", "–")
    c3.metric("♂ Masculino", "–")

# Actividades activas
c4.metric("Actividades activas", df_f[COL_ACT].nunique())

st.markdown("---")

# # =======================
# # Gráfico 1 — Conteo actividad
# # =======================
# st.subheader("Conteo por Actividad / Deporte (total)")
# conteo = df_f[COL_ACT].value_counts().sort_values()
# fig1, ax1 = plt.subplots(figsize=(10, 6))
# conteo.plot(kind="barh", ax=ax1)
# ax1.set_xlabel("Cantidad"); ax1.set_ylabel("Actividad / Deporte")
# ax1.set_title("Cantidad de personas por actividad")
# fig1.tight_layout()
# st.pyplot(fig1); plt.close(fig1)

# # =======================
# # Gráfico 2 — Barras agrupadas (Actividad × Género)
# # =======================
# st.subheader("Actividad Deportiva según el género")
# tabla = pd.crosstab(df_f[COL_ACT], df_f[COL_SEXO])
# fig2, ax2 = plt.subplots(figsize=(12, 6))
# tabla.plot(kind="bar", ax=ax2, width=0.8)
# ax2.set_xlabel("Actividad / Deporte"); ax2.set_ylabel("Cantidad")
# ax2.set_title("Distribución por género")
# plt.xticks(rotation=45, ha="right")
# ax2.legend(title="Sexo")
# fig2.tight_layout()
# st.pyplot(fig2); plt.close(fig2)

# =======================
# Gráfico 3 — 100% apilado
# =======================
st.subheader(" 🏃‍♂️‍➡️Participación porcentual de  Deporte por  género🏃‍♀️")
pct = pd.crosstab(df_f[COL_SEXO], df_f[COL_ACT])
pct = pct.div(pct.sum(axis=1).replace(0, 1), axis=0) * 100
fig3, ax3 = plt.subplots(figsize=(12, 6))
pct.plot(kind="bar", stacked=True, ax=ax3)
ax3.set_xlabel("Género"); ax3.set_ylabel("Porcentaje (%)")
ax3.set_title("Participación porcentual por Actividad / género")
plt.xticks(rotation=0)
ax3.legend(title="Actividad / Deporte", bbox_to_anchor=(1.01, 1), loc="upper left")
fig3.tight_layout()
st.pyplot(fig3); plt.close(fig3)

# # =======================
# # Tabla Top actividades
# # =======================
# st.subheader("🏅 Top actividades")
# top = df_f[COL_ACT].value_counts().reset_index()
# top.columns = [COL_ACT, "Cantidad"]
# st.dataframe(top, use_container_width=True)

# =======================
# Gráfico 4 — IMC histograma
# =======================
if COL_IMC:
    st.subheader("📈 IMC según Género")
    s_imc = pd.to_numeric(df_f[COL_IMC], errors="coerce")
    if s_imc.notna().sum() > 0:
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        for sexo in df_f[COL_SEXO].dropna().unique():
            imc_sexo = s_imc[df_f[COL_SEXO] == sexo]
            ax4.hist(imc_sexo.dropna(), bins=20, alpha=0.5, label=str(sexo))
        ax4.set_xlabel("IMC (kg/m²)"); ax4.set_ylabel("Cantidad de personas")
        ax4.set_title("Distribución de IMC según Género")
        ax4.legend(title="Género")
        fig4.tight_layout()
        st.pyplot(fig4); plt.close(fig4)
    else:
        st.warning("No hay datos numéricos válidos en la columna de IMC.")

# # =======================
# # Gráficos 5 — Boxplots por género (IMC, PAS, PAD, VOP, etc.)
# # =======================
# def try_plot_box(container, col_var, titulo, unidad):
#     if col_var and col_var in df_f.columns:
#         s = pd.to_numeric(df_f[col_var], errors="coerce")
#         if s.notna().sum() > 1:
#             fig = boxplot_por_sexo(df_f.assign(**{col_var: s}), col_var, COL_SEXO, titulo, unidad)
#             container.pyplot(fig); plt.close(fig)
#         else:
#             container.warning(f"No hay suficientes datos numéricos en {col_var} para el boxplot.")

# col_izq, col_der = st.columns(2)
# with col_izq:
#     try_plot_box(st, COL_IMC, "IMC por género", "kg/m²")
#     try_plot_box(st, COL_PAS, "Presión SISTólica por género", "mmHg")
#     try_plot_box(st, COL_CINT, "Perímetro de cintura por género", "cm")
# with col_der:
#     try_plot_box(st, COL_FC,  "Frecuencia cardíaca por género", "lat/min")
#     try_plot_box(st, COL_PAD, "Presión DIASTólica por género", "mmHg")
#     try_plot_box(st, COL_VOP, "Velocidad de Onda del Pulso (VOP) por género", "m/s")
#     try_plot_box(st, COL_PCEN,"Presión central por género", "mmHg")

# =======================
# (Opcional) Histogramas por género de PAS, PAD y VOP
# =======================
def hist_por_genero(df_in, col_num, col_sexo, titulo, unidad=""):
    if not col_num or col_num not in df_in.columns:
        return
    vals = pd.to_numeric(df_in[col_num], errors="coerce")
    ok = vals.notna() & (~df_in[col_sexo].isna())
    if ok.sum() < 2:
        st.info(f"No hay suficientes datos numéricos en {col_num} para el histograma.")
        return
    dfh = df_in.loc[ok, [col_num, col_sexo]].copy()
    vmin = float(dfh[col_num].min()); vmax = float(dfh[col_num].max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0, 1
    bins = np.linspace(vmin, vmax, 24)

    fig, ax = plt.subplots(figsize=(9,5))
    for sexo in dfh[col_sexo].astype(str).unique():
        x = dfh.loc[dfh[col_sexo].astype(str)==sexo, col_num].dropna().values
        if len(x)==0: continue
        ax.hist(x, bins=bins, alpha=0.45, label=f"{sexo} (n={len(x)})")
        med = np.median(x)
        ax.axvline(med, linestyle="--", alpha=0.7)
        ax.text(med, ax.get_ylim()[1]*0.9, f"Med {sexo}={med:.1f}", rotation=90,
                va="top", ha="right", fontsize=9)

    ax.set_xlabel(f"{col_num}" + (f" ({unidad})" if unidad else ""))
    ax.set_ylabel("Cantidad")
    ax.set_title(titulo)
    ax.legend(title="Género")
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

if mostrar_hist:
    st.subheader("📊 Presión Arterial Sistólica por género")
    hist_por_genero(df_f, COL_PAS, COL_SEXO, "PAS por género", "mmHg")
    st.subheader("📊 Presión Arterial Diastólica por género")
    hist_por_genero(df_f, COL_PAD, COL_SEXO, "PAD por género", "mmHg")
    # st.subheader("📊 Velocidad de Onda del Pulso (VOP) por género")
    # def vop_violin(df, col_vop, col_sexo, titulo="VOP por sexo (violín)", unidad="m/s"):
    #     if col_vop not in df.columns or col_sexo not in df.columns:
    #         st.warning("Columnas faltantes.")
    #         return

    #     df = df[[col_vop, col_sexo]].dropna()
    #     if len(df) == 0:
    #         st.info("No hay datos de VOP.")
    #         return

    #     df[col_vop] = pd.to_numeric(df[col_vop], errors="coerce")
    #     df = df.dropna(subset=[col_vop])
    #     df[col_sexo] = df[col_sexo].astype(str)

    #     grupos = [g for _, g in df.groupby(df[col_sexo])]
    #     labels = [str(g[col_sexo].iloc[0]) for g in grupos]
    #     data = [g[col_vop].values for g in grupos]

    #     fig, ax = plt.subplots(figsize=(7,4))
    #     if len(df) < 3:
    #         # con pocos datos, uso scatter en vez de violín
    #         for i, (lab, vals) in enumerate(zip(labels, data), start=1):
    #             ax.scatter([i]*len(vals), vals, s=80, label=f"{lab} (n={len(vals)})")
    #         ax.legend()
    #     else:
    #         ax.violinplot(data, showmedians=True, showextrema=False)
    #         ax.set_xticks(range(1, len(labels)+1))
    #         ax.set_xticklabels(labels)
    #     ax.set_ylabel(f"{col_vop} ({unidad})")
    #     ax.set_title(titulo)
    #     ax.grid(axis="y", alpha=0.3)
    #     st.pyplot(fig); plt.close(fig)


    #     #  LLAMo FUNCION 
    #     vop_violin(df_f, COL_VOP, COL_SEXO, "Velocidad de Onda del Pulso (VOP) por género", "m/s")

        

# =======================
# Gráfico 6 — Antecedentes/Patología 
# =======================
import re, unicodedata, numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Helpers de normalización y parsing ---
def _norm(s: str) -> str:
    return unicodedata.normalize("NFD", s).encode("ascii","ignore").decode("utf-8").strip().lower()

_SPLIT_RE = re.compile(r"[;,/]+|\s{2,}")

def _split_multi(cell) -> list[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = _norm(str(cell))
    # separa por ; , / o múltiples espacios, y también admite coma simple
    parts = re.split(r"[;,/]", s)
    # limpiar espacios y vacíos
    return [p.strip() for p in parts if p.strip()]

# --- Diccionario de opciones normalizadas ---
# (si tus celdas vienen con estos textos, se reconocerán exacto)
OPC_ANT = {
    _norm(x) for x in [
        "Hipo/Hipertiroidismo","Sobrepeso u obesidad","Diabetes","migrana",
        "dislipidemia","enfermedad renal","infarto","ninguna"
    ]
}
OPC_PATO = {
    _norm(x) for x in [
        "dislipidemia","Hipo/Hipertiroidismo","Consumo de tabaco",
        "Consumo de psicofármacos","Problemas respiratorios (Asma Bronquial, otros)",
        "enfermedad renal","consumo de alcohol","ninguna"
    ]
}

# Sinónimos/keywords por condición (buscados en texto libre)
TOK_HTA   = ["hta","hipert","tension","presion alta","hiper tenso"]
TOK_COLE  = ["colest","dislip","ldl","hdl bajo","hipercoles"]
TOK_OBES  = ["obes","sobrepeso","imc alto"]
TOK_TAB   = ["taba","fum","cigarr","exfum"]
TOK_DIAB  = ["diab","dm","gluc","hipergluc","azucar alta"]
TOK_TIRO  = ["hipotiro","hipertiro","tiroid"]
TOK_RENAL = ["renal","nefrop"]
TOK_RESP  = ["asma","respir", "epoc","bronqu"]

def _text_has_any(text: str, tokens: list[str]) -> bool:
    return any(tok in text for tok in tokens)

# --- Extrae "sets" de opciones por fila (exact match) y texto libre combinado ---
cols_txt = [COL_ANT, COL_PATO]  # tus columnas
def _row_sets_and_text(row):
    ant_vals = set(_split_multi(row[COL_ANT])) if COL_ANT in row.index else set()
    pat_vals = set(_split_multi(row[COL_PATO])) if COL_PATO in row.index else set()
    # Texto libre (por si escribieron algo más)
    parts = []
    for c in cols_txt:
        if c in row.index:
            v = row[c]
            if pd.notna(v) and str(v).strip() != "":
                parts.append(str(v))
    txt = _norm(" ".join(parts))
    return ant_vals, pat_vals, txt

# --- Columnas numéricas (si existen) ---
has_pas = COL_PAS and COL_PAS in df_f.columns
has_pad = COL_PAD and COL_PAD in df_f.columns
has_imc = COL_IMC and COL_IMC in df_f.columns

PAS = pd.to_numeric(df_f[COL_PAS], errors="coerce") if has_pas else pd.Series([pd.NA]*len(df_f), index=df_f.index)
PAD = pd.to_numeric(df_f[COL_PAD], errors="coerce") if has_pad else pd.Series([pd.NA]*len(df_f), index=df_f.index)
IMC = pd.to_numeric(df_f[COL_IMC], errors="coerce") if has_imc else pd.Series([pd.NA]*len(df_f), index=df_f.index)

# --- Detectores por condición (usa: opciones exactas + texto + números) ---
def _flag_known_HTA(idx, ant, pat, txt):
    # Texto/keywords o números (PAS/PAD)
    has_token = _text_has_any(txt, TOK_HTA)
    pasv = PAS.loc[idx]; padv = PAD.loc[idx]
    has_num = (pd.notna(pasv) and pasv >= 140) or (pd.notna(padv) and padv >= 90)
    flag = bool(has_token or has_num or ("hta" in ant) or ("hta" in pat) or ("hipertension" in ant) or ("hipertension" in pat))
    known = bool(len(ant|pat) > 0 or pd.notna(pasv) or pd.notna(padv) or len(txt) > 0)
    return flag, known

def _flag_known_OBES(idx, ant, pat, txt):
    has_token = _text_has_any(txt, TOK_OBES) or ("sobrepeso u obesidad" in ant) or ("sobrepeso u obesidad" in pat) or ("obesidad" in ant) or ("obesidad" in pat)
    imcv = IMC.loc[idx]
    has_num = (pd.notna(imcv) and imcv >= 30)
    flag = bool(has_token or has_num)
    known = bool(len(ant|pat) > 0 or pd.notna(imcv) or len(txt) > 0)
    return flag, known

def _flag_known_COLE(idx, ant, pat, txt):
    has_token = _text_has_any(txt, TOK_COLE) or ("dislipidemia" in ant) or ("dislipidemia" in pat) or ("colesterol" in ant) or ("colesterol" in pat)
    known = bool(len(ant|pat) > 0 or len(txt) > 0)
    return bool(has_token), known

def _flag_known_TAB(idx, ant, pat, txt):
    has_token = _text_has_any(txt, TOK_TAB) or ("consumo de tabaco" in pat) or ("tabaco" in ant) or ("tabaco" in pat) or ("tabaquismo" in ant) or ("tabaquismo" in pat)
    known = bool(len(ant|pat) > 0 or len(txt) > 0)
    return bool(has_token), known

def _flag_known_DIAB(idx, ant, pat, txt):
    has_token = _text_has_any(txt, TOK_DIAB) or ("diabetes" in ant) or ("diabetes" in pat)
    known = bool(len(ant|pat) > 0 or len(txt) > 0)
    return bool(has_token), known

def _flag_known_TIRO(idx, ant, pat, txt):
    has_token = _text_has_any(txt, TOK_TIRO) or ("hipo/hipertiroidismo" in ant) or ("hipo/hipertiroidismo" in pat)
    known = bool(len(ant|pat) > 0 or len(txt) > 0)
    return bool(has_token), known

def _flag_known_RENAL(idx, ant, pat, txt):
    has_token = _text_has_any(txt, TOK_RENAL) or ("enfermedad renal" in ant) or ("enfermedad renal" in pat)
    known = bool(len(ant|pat) > 0 or len(txt) > 0)
    return bool(has_token), known

def _flag_known_RESP(idx, ant, pat, txt):
    has_token = _text_has_any(txt, TOK_RESP) or ("problemas respiratorios (asma bronquial, otros)" in pat)
    known = bool(len(ant|pat) > 0 or len(txt) > 0)
    return bool(has_token), known

# ——— Panel de donas (puedes dejar tu función _dona igual) ———
def _dona(container, porcentaje, titulo, subt=""):
    fig, ax = plt.subplots(figsize=(2.9, 2.9))
    val = max(0.0, min(100.0, porcentaje))
    ax.pie([val, 100 - val], startangle=90, colors=["#D72638", "#E8E8E8"], wedgeprops={'width': 0.25})
    ax.text(0, 0, f"{val:.1f}%", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.set_title(titulo + (f"\n{subt}" if subt else ""), fontsize=10, pad=8)
    ax.axis("equal")
    fig.tight_layout()
    container.pyplot(fig); plt.close(fig)

# ——— Mapeo de condiciones a funciones ———
COND_FUNCS = {
    "Hipertensión Arterial": _flag_known_HTA,
    "Hipercolesterolemia / Dislipidemia": _flag_known_COLE,
    "Obesidad / Sobrepeso": _flag_known_OBES,
    "Tabaquismo": _flag_known_TAB,
    "Glucemia elevada / Diabetes": _flag_known_DIAB,
    "Trastornos tiroideos": _flag_known_TIRO,
    "Enfermedad renal": _flag_known_RENAL,
    "Problemas respiratorios": _flag_known_RESP,
}

# ——— Loop por género y cálculo de % válidos ———
st.subheader("🩺 Proporción de patologías por género (todas las condiciones)")
sexos_presentes = list(df_f[COL_SEXO].dropna().astype(str).unique())
if len(sexos_presentes) == 0:
    st.info("No hay datos de género para estas donas.")
else:
    for sexo in sexos_presentes:
        st.markdown(f"**Género: {sexo}**")
        sub_idx = df_f[COL_SEXO].astype(str) == sexo

        resultados = {}
        for nombre, fn in COND_FUNCS.items():
            flags, knowns = [], []
            for idx, row in df_f[sub_idx].iterrows():
                ant_set, pat_set, txt = _row_sets_and_text(row)
                f, k = fn(idx, ant_set, pat_set, txt)
                flags.append(bool(f)); knowns.append(bool(k))
            known_n = int(np.sum(knowns))
            if known_n == 0:
                resultados[nombre] = (None, 0)
            else:
                pct = 100.0 * (np.sum(flags) / known_n)
                resultados[nombre] = (pct, known_n)

        cols = st.columns(3)
        for i, nombre in enumerate(list(COND_FUNCS.keys())):
            c = cols[i % 3]
            pct, known_n = resultados[nombre]
            if pct is None:
                with c: st.info(f"{nombre}: sin datos suficientes.")
            else:
                _dona(c, pct, nombre, subt=f"válidos n={known_n}")
    

st.markdown("---")
st.caption("Desarrollado por GIBIO FRBA — Streamlit Dashboard © 2025")
