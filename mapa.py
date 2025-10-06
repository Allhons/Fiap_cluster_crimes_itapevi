# app_streamlit_heatmap_itapevi_maior.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ---------------------------
# CONFIGURAÇÕES
# ---------------------------
DATA_PATH = r"C:\Users\Allyson\Documents\ML- seguranca_pos\data\SPDadosCriminais_Itapevi(2025).xlsx"
MODEL_PATH = r"models_output/best_kmeans_score0.9999.joblib"

LAT_COL = "LATITUDE"
LON_COL = "LONGITUDE"
DATA_COL = "DATA_OCORRENCIA_BO"
HORA_COL = "HORA_OCORRENCIA_BO"

# Coordenadas fixas do centro de Itapevi
ITAPEVI_CENTER = [-23.5506508472123, -46.93916987719007]

# ---------------------------
# CARREGAR DADOS
# ---------------------------
@st.cache_data
def load_data(path):
    df = pd.read_excel(path, engine="openpyxl")
    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
    df = df.dropna(subset=[LAT_COL, LON_COL])
    
    df[DATA_COL] = pd.to_datetime(df[DATA_COL], errors="coerce", dayfirst=True)
    df["weekday"] = df[DATA_COL].dt.dayofweek
    
    def parse_hour(v):
        if pd.isna(v):
            return np.nan
        try:
            if ":" in str(v):
                return int(str(v).split(":")[0])
            return int(float(v))
        except:
            return np.nan
    df["hour"] = df[HORA_COL].apply(parse_hour)
    
    return df

df = load_data(DATA_PATH)

# ---------------------------
# CARREGAR MODELO TREINADO
# ---------------------------
@st.cache_resource
def load_model(path):
    loaded = joblib.load(path)
    return loaded["model"]

model = load_model(MODEL_PATH)

# ---------------------------
# TÍTULO STREAMLIT
# ---------------------------
st.set_page_config(page_title="Mapa de Calor - Itapevi", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #d7263d; font-size: 3em; font-weight: bold;'>
        🔥 Mapa de Calor de Crimes - Itapevi 🔥
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown("<div style='text-align: center;'>Visualize a densidade de crimes filtrando por dia da semana e horário.</div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: #22223b; font-size: 1.5em; font-weight: bold; margin-bottom: 1em;'>
        🚨 Explore os pontos mais críticos de criminalidade em Itapevi! <br>
        Filtre por dia, horário, bairro, tipo de crime e rubrica para identificar padrões e tomar decisões estratégicas.<br>
        <span style='color:#d7263d;'>Segurança começa com informação!</span>
    </div>
    """,
    unsafe_allow_html=True
)
# ---------------------------
# FILTROS MELHORADOS NA SIDEBAR
# ---------------------------
with st.sidebar:
    st.header("Filtros avançados")
    dias_semana = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
    dia_filtro = st.multiselect("Selecione o(s) dia(s) da semana:", options=dias_semana, default=dias_semana)
    horario_filtro = st.slider("Selecione o intervalo de horário:", 0, 23, (0, 23))

    # Filtro por intervalo de datas
    data_min = df[DATA_COL].min().date()
    data_max = df[DATA_COL].max().date()
    data_range = st.date_input("Selecione o intervalo de datas:", (data_min, data_max), min_value=data_min, max_value=data_max)

    # Filtro por bairro (se existir)
    bairro_filtro = None
    if "BAIRRO" in df.columns:
        bairros = df["BAIRRO"].dropna().unique().tolist()
        bairros.sort()
        bairro_filtro = st.multiselect("Selecione o(s) bairro(s):", options=bairros, default=bairros)

    # Filtro por tipo de crime (se existir)
    tipo_filtro = None
    if "TIPO_CRIME" in df.columns:
        tipos = df["TIPO_CRIME"].dropna().unique().tolist()
        tipos = [t for t in tipos if t not in ["Estupro - Art. 213", "Estupro de vulneravel (art.217-A)"]]
        tipos.sort()
        tipo_filtro = st.multiselect("Selecione o(s) tipo(s) de crime:", options=tipos, default=tipos)

    # Filtro por rubrica (se existir)
    rubrica_filtro = None
    if "RUBRICA" in df.columns:
        rubricas = df["RUBRICA"].dropna().unique().tolist()
        rubricas.sort()
        rubrica_filtro = st.multiselect("Selecione o(s) tipo(s) de rubrica:", options=rubricas, default=rubricas)

# Criação do df_filtrado inicial
dias_dict = {v: k for k, v in enumerate(dias_semana)}
dias_numeros = [dias_dict[d] for d in dia_filtro]

df_filtrado = df[
    (df["weekday"].isin(dias_numeros)) &
    (df["hour"] >= horario_filtro[0]) & (df["hour"] <= horario_filtro[1]) &
    (df[DATA_COL].dt.date >= data_range[0]) & (df[DATA_COL].dt.date <= data_range[1])
]

if bairro_filtro is not None:
    df_filtrado = df_filtrado[df_filtrado["BAIRRO"].isin(bairro_filtro)]

if tipo_filtro is not None:
    df_filtrado = df_filtrado[df_filtrado["TIPO_CRIME"].isin(tipo_filtro)]

if rubrica_filtro is not None:
    df_filtrado = df_filtrado[df_filtrado["RUBRICA"].isin(rubrica_filtro)]

#st.write(f"Registros filtrados: {len(df_filtrado)}")


# ---------------------------
# PREDIÇÃO DE CLUSTERS
# ---------------------------
X = df_filtrado[[LAT_COL, LON_COL]].to_numpy(dtype=float)
labels = model.named_steps[list(model.named_steps.keys())[-1]].predict(X)
df_filtrado["cluster"] = labels.astype(str)

# ---------------------------
# MAPA DE CALOR FOLIUM CENTRALIZADO EM ITAPEVI
# ---------------------------
m = folium.Map(location=ITAPEVI_CENTER, zoom_start=13)
HeatMap(df_filtrado[[LAT_COL, LON_COL]].values.tolist(), radius=15, blur=20, max_zoom=13).add_to(m)

# AUMENTAR O TAMANHO DO MAPA
st_folium(m, width="100%", height=800)
