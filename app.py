import streamlit as st
import pandas as pd
import os
from datetime import datetime, timezone

# Configuración de la página
st.set_page_config(page_title="TimesFM Crypto Predictor", page_icon="🧠", layout="wide")

st.title("🧠 Dashboard Predictivo Cripto (TimesFM-3)")
st.markdown("Proyecciones a 12 horas impulsadas por el modelo fundacional de series temporales de Google Research.")

# Verificamos que el archivo exista Y que no esté vacío (mayor a 0 bytes)
if os.path.exists("predicciones.csv") and os.path.getsize("predicciones.csv") > 0:
    try:
        df = pd.read_csv("predicciones.csv")
        
        # Mostrar la fecha de la última actualización
        fecha_mod = datetime.fromtimestamp(os.path.getmtime("predicciones.csv"), tz=timezone.utc)
        st.caption(f"Última actualización de datos: {fecha_mod.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        st.subheader("Top Activos")
        cols = st.columns(3)
        for i, col in enumerate(cols):
            if i < len(df):
                ticker = str(df.iloc[i]['ticker']).replace('USDT', '')
                precio = float(df.iloc[i]['precio'])
                var = float(df.iloc[i]['variacion'])
                senal = str(df.iloc[i]['senal'])
                
                col.metric(
                    label=f"{ticker} ({senal})", 
                    value=f"${precio:.4f}", 
                    delta=f"{var:.2f}% a 12h"
                )
                
        st.divider()
        st.subheader("Tabla Completa de Predicciones")
        
        # Formatear datos para la tabla
        df['ticker'] = df['ticker'].astype(str).str.replace('USDT', '')
        df['precio'] = df['precio'].apply(lambda x: f"${float(x):,.4f}")
        df['proyectado'] = df['proyectado'].apply(lambda x: f"${float(x):,.4f}")
        df['variacion'] = df['variacion'].apply(lambda x: f"{float(x):,.2f}%")
        
        st.dataframe(df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Hubo un error al leer los datos. El Bot los está actualizando. (Error: {e})")
else:
    st.warning("⚠️ Los datos aún no se han generado o el archivo está vacío. Ve a GitHub Actions y ejecuta el bot.")
