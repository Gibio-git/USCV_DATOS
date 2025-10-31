import streamlit as st  # framework web
import pandas as pd     # manejo de dataframes
import numpy as np
import matplotlib.pyplot as plt  # gráficos
import unicodedata, re            # normalización de textos

# ===================== Configuración general =====================
st.set_page_config(
    page_title="Protocolo GIBIO Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== Vista compacta (menos scroll) =====
FIG_W, FIG_H = 5.2, 3.2   # tamaño de figura en pulgadas

st.markdown("""
<style>
div.block-container {padding-top: 0.35rem; padding-bottom: 0.2rem; max-width: 1600px;}
section[data-testid="stVerticalBlock"] {gap: 0.40rem;}
h1,h2,h3,h4,h5 {margin: 0.15rem 0 0.15rem 0;}
</style>
""", unsafe_allow_html=True)

import matplotlib as mpl
mpl.rcParams.update({
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.autolayout": True,
})

def make_fig():
    """Figura pequeña para Streamlit."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)
    return fig, ax

# ===================== Helpers =====================
def clean_headers(cols):  # normaliza y limpia encabezados
    return [
        re.sub(r"\s+", " ",
               unicodedata.normalize("NFD", c).encode("ascii", "ignore").decode("utf-8")
        ).strip().upper()
        for c in cols
    ]

def pick_col(df, rules):
    """
    Devuelve el nombre de la primera columna cuyo encabezado (upper)
    contenga TODOS los tokens de una de las reglas.
    """
    for r in rules:
        for c in df.columns:
            name = c.upper()
            if all(tok in name for tok in r):
                return c
    return None

def to_number(series: pd.Series) -> pd.Series:
    """Convierte textos como '12,3', '120 mmHg', '10 m/s' a float. Si no hay número, NaN."""
    s = series.astype(str).str.replace(",", ".", regex=False)
    s = s.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(s, errors="coerce")

# ======================= Carga de datos =======================
df = pd.read_excel("EVALUACIONES2024.xlsx", header=1)  # encabezados reales en fila 2 (0-index)
df.columns = clean_headers(df.columns)

# ======================= Detección flexible de columnas =======================
COL_ACT  = pick_col(df, [["ACTIVIDAD","DEPOR"], ["DEPORTE"], ["ACTIVIDAD"]])
COL_SEXO = pick_col(df, [["SEXO"], ["GENERO"]])
COL_FC   = pick_col(df, [["FREC","CARDI"], ["FC 1"], ["FC"]])
COL_PAS  = pick_col(df, [["PAS 1"], ["PAS"], ["SISTO"]])
COL_PAD  = pick_col(df, [["PAD 1"], ["PAD"], ["DIASTO"]])
COL_CINT = pick_col(df, [["CINTUR"]])
COL_PCEN = pick_col(df, [["PRESION","CENTRAL"], ["PCENTRAL"], ["PRES CENT"]])
COL_ANT  = pick_col(df, [["ANTECEDENTES","CV"], ["ANTECEDENTES CV"], ["ANTECEDENTES"]])
COL_PESO = pick_col(df, [["PESO"]])
COL_ALT  = pick_col(df, [["ALTURA"]])
COL_IMC  = pick_col(df, [["IMC"]])
COL_PATO = pick_col(df, [["PATOLOG"], ["PREEXIST"], ["ENFERM"]])

# # ======================= Título y validación =======================
# st.title("📊 Protocolo de Evaluaciones Cardiovasculares — GIBIO FRBA")

# ======== Título centrado con espacio superior ========
st.markdown("""
<div style="
    text-align: center;
    font-size: 30px;
    font-weight: 700;
    line-height: 1.3;
    white-space: normal;
    overflow-wrap: break-word;
    word-break: break-word;
    margin-top: 2.5rem;        /* 🔹 más margen arriba */
    margin-bottom: 1rem;       /* 🔹 mantiene buena separación abajo */
">
📊 Protocolo de Evaluaciones Cardiovasculares — 
<span style="color:#58a6ff;">GIBIO FRBA</span>
</div>
""", unsafe_allow_html=True)


if not COL_ACT or not COL_SEXO:
    st.error(f"No encontré columnas de ACTIVIDAD/DEPORTE o SEXO/GÉNERO.\nEncabezados: {list(df.columns)}")
    st.stop()

# ======================= Sidebar filtros =======================
with st.sidebar:
    st.header("Filtros")
    actividades = sorted([a for a in df[COL_ACT].dropna().astype(str).str.strip().unique() if a!=""])
    sexos       = sorted([s for s in df[COL_SEXO].dropna().astype(str).str.strip().unique() if s!=""])
    f_sexo = st.multiselect("Sexo / Género", options=sexos, default=sexos)
    f_act  = st.multiselect("Actividad / Deporte", options=actividades, default=actividades)
    q = st.text_input("Buscar actividad (contiene)", "")
    mostrar_hist = st.checkbox("Mostrar PAS y PAD (abajo)", value=True)

# Aplicar filtros
df_f = df.copy()
if f_sexo: df_f = df_f[df_f[COL_SEXO].isin(f_sexo)]
if f_act:  df_f = df_f[df_f[COL_ACT].isin(f_act)]
if q.strip():
    df_f = df_f[df_f[COL_ACT].str.contains(q.strip(), case=False, regex=False)]

# ======================= Normalización numérica =======================
for col in [COL_PAS, COL_PAD, COL_PCEN, COL_FC, COL_CINT, COL_IMC, COL_PESO, COL_ALT]:
    if col and col in df.columns:
        df[col] = to_number(df[col])
    if col and col in df_f.columns:
        df_f[col] = to_number(df_f[col])

# ======================= KPIs =======================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros", f"{len(df_f):,}".replace(",", "."))
if COL_SEXO in df_f.columns:
    sex_counts = df_f[COL_SEXO].astype(str).str.upper().value_counts()
    fem = sex_counts.get("F", 0)
    masc = sex_counts.get("M", 0)
    otros = sex_counts.sum() - fem - masc
    c2.metric("♀ Femenino (F)", fem)
    c3.metric("♂ Masculino (M)", masc)
    if otros > 0:
        st.caption(f"Otros / sin datos: {otros}")
else:
    c2.metric("♀ Femenino", "–")
    c3.metric("♂ Masculino", "–")
c4.metric("Actividades activas", df_f[COL_ACT].nunique())

st.markdown("---")

# ======================= Grilla superior: Actividad % (izq) e IMC (der) =======================
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**🏃‍♂️‍➡️Participación porcentual de  Deporte por  género🏃‍♀️**")
    pct = pd.crosstab(df_f[COL_SEXO], df_f[COL_ACT])
    pct = pct.div(pct.sum(axis=1).replace(0, 1), axis=0) * 100
    fig_pct, ax_pct = make_fig()
    pct.plot(kind="bar", stacked=True, ax=ax_pct, legend=False)
    ax_pct.set_xlabel("Género"); ax_pct.set_ylabel("Porcentaje (%)")
    st.pyplot(fig_pct); plt.close(fig_pct)

with col_right:
    st.markdown("**📈 IMC según Género**")
    if COL_IMC:
        s_imc = pd.to_numeric(df_f[COL_IMC], errors="coerce")
        if s_imc.notna().sum() > 0:
            fig_imc, ax_imc = make_fig()
            for sexo in df_f[COL_SEXO].dropna().unique():
                imc_sexo = s_imc[df_f[COL_SEXO] == sexo]
                ax_imc.hist(imc_sexo.dropna(), bins=18, alpha=0.55, label=str(sexo))
            ax_imc.set_xlabel("IMC (kg/m²)"); ax_imc.set_ylabel("Cantidad")
            ax_imc.legend(frameon=False, title="Género")
            st.pyplot(fig_imc); plt.close(fig_imc)
        else:
            st.caption("Sin datos numéricos en IMC.")
    else:
        st.caption("Columna IMC no detectada.")

# ======================= Grilla inferior: PAS y PAD (separados) =======================
def hist_por_genero_compacto(df_in, col_num, col_sexo, unidad=""):
    if not col_num or col_num not in df_in.columns:
        st.caption("Columna faltante."); return
    vals = pd.to_numeric(df_in[col_num], errors="coerce")
    ok = vals.notna() & (~df_in[col_sexo].isna())
    if ok.sum() < 2:
        st.caption(f"Sin datos suficientes en {col_num}."); return

    dfh = df_in.loc[ok, [col_num, col_sexo]].copy()
    vmin = float(dfh[col_num].min()); vmax = float(dfh[col_num].max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0, 1
    bins = np.linspace(vmin, vmax, 16)  # menos bins → más compacto

    fig, ax = make_fig()
    for sexo in dfh[col_sexo].astype(str).unique():
        x = dfh.loc[dfh[col_sexo].astype(str)==sexo, col_num].dropna().values
        if len(x)==0: continue
        ax.hist(x, bins=bins, alpha=0.45, label=f"{sexo} (n={len(x)})")
        med = np.median(x)
        ax.axvline(med, linestyle="--", alpha=0.7)
    ax.set_xlabel(f"{col_num}" + (f" ({unidad})" if unidad else ""))
    ax.set_ylabel("Cantidad")
    ax.legend(frameon=False, ncol=2)
    st.pyplot(fig); plt.close(fig)

if mostrar_hist:
    st.markdown("---")
    col_pas, col_pad = st.columns(2)
    with col_pas:
        st.markdown("**📊 Presión Arterial Sistólica por género**")
        hist_por_genero_compacto(df_f, COL_PAS, COL_SEXO, "mmHg")
    with col_pad:
        st.markdown("**📊 Presión Arterial Diastólica por género**")
        hist_por_genero_compacto(df_f, COL_PAD, COL_SEXO, "mmHg")

# ======================= Panel de donas por patologías (se mantiene) =======================
# --- Normalizadores y detectores ---
def _norm(s: str) -> str:
    return unicodedata.normalize("NFD", s).encode("ascii","ignore").decode("utf-8").strip().lower()

def _split_multi(cell) -> list[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)): return []
    s = _norm(str(cell))
    parts = re.split(r"[;,/]", s)
    return [p.strip() for p in parts if p.strip()]

TOK_HTA   = ["hta","hipert","tension","presion alta","hiper tenso"]
TOK_COLE  = ["colest","dislip","ldl","hdl bajo","hipercoles"]
TOK_OBES  = ["obes","sobrepeso","imc alto"]
TOK_TAB   = ["taba","fum","cigarr","exfum"]
TOK_DIAB  = ["diab","dm","gluc","hipergluc","azucar alta"]
TOK_TIRO  = ["hipotiro","hipertiro","tiroid"]
TOK_RENAL = ["renal","nefrop"]
TOK_RESP  = ["asma","respir","epoc","bronqu"]

cols_txt = [COL_ANT, COL_PATO]
def _row_sets_and_text(row):
    ant_vals = set(_split_multi(row[COL_ANT])) if COL_ANT in row.index else set()
    pat_vals = set(_split_multi(row[COL_PATO])) if COL_PATO in row.index else set()
    parts = []
    for c in cols_txt:
        if c in row.index:
            v = row[c]
            if pd.notna(v) and str(v).strip() != "": parts.append(str(v))
    txt = _norm(" ".join(parts))
    return ant_vals, pat_vals, txt

has_pas = COL_PAS and COL_PAS in df_f.columns
has_pad = COL_PAD and COL_PAD in df_f.columns
has_imc = COL_IMC and COL_IMC in df_f.columns

PAS = pd.to_numeric(df_f[COL_PAS], errors="coerce") if has_pas else pd.Series([pd.NA]*len(df_f), index=df_f.index)
PAD = pd.to_numeric(df_f[COL_PAD], errors="coerce") if has_pad else pd.Series([pd.NA]*len(df_f), index=df_f.index)
IMC = pd.to_numeric(df_f[COL_IMC], errors="coerce") if has_imc else pd.Series([pd.NA]*len(df_f), index=df_f.index)

def _text_has_any(text: str, tokens: list[str]) -> bool:
    return any(tok in text for tok in tokens)

def _flag_known_HTA(idx, ant, pat, txt):
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

def _dona(container, porcentaje, titulo, subt=""):
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    val = max(0.0, min(100.0, porcentaje))
    ax.pie([val, 100 - val], startangle=90, colors=["#D72638", "#E8E8E8"], wedgeprops={'width': 0.25})
    ax.text(0, 0, f"{val:.1f}%", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_title(titulo + (f"\n{subt}" if subt else ""), fontsize=9, pad=6)
    ax.axis("equal")
    fig.tight_layout(pad=0.2)
    container.pyplot(fig); plt.close(fig)

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
