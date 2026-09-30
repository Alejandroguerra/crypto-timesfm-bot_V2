import streamlit as st
import pandas as pd
import os
from datetime import datetime, timezone

# Configuración de la página
st.set_page_config(page_title="TimesFM Crypto Predictor", page_icon="🧠", layout="wide")

st.title("🧠 Dashboard Predictivo Cripto (TimesFM-3)")
st.markdown("Proyecciones a 12 horas impulsadas por el modelo fundacional de series temporales de Google Research.")

# Leer el archivo generado por nuestro bot
if os.path.exists("predicciones.csv"):
    df = pd.read_csv("predicciones.csv")
    
    # Mostrar la fecha de la última actualización de los datos
    fecha_mod = datetime.fromtimestamp(os.path.getmtime("predicciones.csv"), tz=timezone.utc)
    st.caption(f"Última actualización de datos: {fecha_mod.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    # Crear métricas visuales para las top 3 monedas (BTC, ETH, SOL)
    st.subheader("Top Activos")
    cols = st.columns(3)
    for i, col in enumerate(cols):
        if i < len(df):
            ticker = df.iloc[i]['ticker'].replace('USDT', '')
            precio = round(df.iloc[i]['precio'], 2)
            var = df.iloc[i]['variacion']
            senal = df.iloc[i]['senal']
            
            col.metric(
                label=f"{ticker} ({senal})", 
                value=f"${precio}", 
                delta=f"{var:.2f}% a 12h"
            )
            
    st.divider()
    
    # Mostrar la tabla completa interactiva
    st.subheader("Tabla Completa de Predicciones")
    
    # Limpiar un poco para mostrar en pantalla
    df['ticker'] = df['ticker'].str.replace('USDT', '')
    df['precio'] = df['precio'].apply(lambda x: f"${x:,.4f}")
    df['proyectado'] = df['proyectado'].apply(lambda x: f"${x:,.4f}")
    df['variacion'] = df['variacion'].apply(lambda x: f"{x:,.2f}%")
    
    st.dataframe(df, use_container_width=True)
else:
    st.warning("⚠️ Los datos aún no se han generado. El bot de GitHub Actions está calculando las predicciones. Por favor vuelve en unos minutos.")
