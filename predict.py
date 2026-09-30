import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import timesfm
import yfinance as yf

# En Yahoo Finance, los pares se buscan con -USD
TICKERS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", 
    "ADA-USD", "AVAX-USD", "LINK-USD", "DOT-USD", "NEAR-USD"
]
CONTEXT_LEN = 512 
HORIZON_LEN = 12  

def inicializar_modelo_timesfm():
    print("Cargando modelo TimesFM desde Hugging Face...")
    tfm = timesfm.TimesFm(
        context_len=CONTEXT_LEN,
        horizon_len=HORIZON_LEN,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend="cpu" 
    )
    tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
    print("✅ Modelo cargado correctamente.")
    return tfm

def obtener_datos_yahoo(symbol, limit=CONTEXT_LEN):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1mo", interval="1h")
        if len(df) < limit:
            return None
        
        # Tomamos exactamente las últimas 512 horas
        df = df.tail(limit)
        df['close'] = df['Close'].astype(float)
        return df
    except Exception as e:
        print(f"Error descargando {symbol}: {e}")
        return None

def predecir_con_timesfm(tfm, df):
    historia_precios = df['close'].values
    precio_actual = historia_precios[-1]

    # Inferencia con TimesFM
    forecast_result = tfm.forecast(inputs=[historia_precios], freq=[0])
    predicciones_futuras = forecast_result[0][0] 
    precio_proyectado_12h = predicciones_futuras[-1] 
    
    variacion_pct = ((precio_proyectado_12h - precio_actual) / precio_actual) * 100
    
    if variacion_pct > 1.5:
        senal = "COMPRA 🟢"
    elif variacion_pct < -1.5:
        senal = "VENTA 🔴"
    else:
        senal = "MANTENER ⚪"
        
    return senal, precio_actual, variacion_pct, precio_proyectado_12h

def actualizar_readme(resultados):
    fecha_actual = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    md = f"# 🧠 TimesFM Crypto Predictor\n\n"
    md += f"**Última actualización:** `{fecha_actual}`\n\n"
    md += "> Predicciones generadas utilizando el modelo fundacional *TimesFM* de Google Research. Horizonte de predicción: 12 horas.\n\n"
    
    md += "| Criptomoneda | Precio Actual | Proyección (12h) | Variación Est. | Señal Sugerida |\n"
    md += "| :--- | :--- | :--- | :--- | :--- |\n"
    
    for res in resultados:
        precio = f"${res['precio']:.4f}".rstrip('0').rstrip('.')
        proy = f"${res['proyectado']:.4f}".rstrip('0').rstrip('.')
        var = f"{res['variacion']:.2f}%"
        
        if res['variacion'] > 0: var = f"+{var} 📈"
        elif res['variacion'] < 0: var = f"{var} 📉"
            
        md += f"| **{res['ticker']}** | {precio} | {proy} | {var} | **{res['senal']}** |\n"

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("✅ README.md actualizado.")

def main():
    tfm = inicializar_modelo_timesfm()
    resultados = []
    
    for ticker in TICKERS:
        df = obtener_datos_yahoo(ticker, limit=CONTEXT_LEN)
        
        if df is not None and len(df) == CONTEXT_LEN:
            senal, precio, variacion, proyectado = predecir_con_timesfm(tfm, df)
            
            # Guardamos el nombre limpio (ej. "BTC" en vez de "BTC-USD") para que Streamlit lo lea fácil
            nombre_limpio = ticker.replace('-USD', '')
            
            resultados.append({
                "ticker": nombre_limpio,
                "precio": precio,
                "proyectado": proyectado,
                "variacion": variacion,
                "senal": senal
            })
            print(f"{nombre_limpio}: Procesado ({senal})")
        else:
            print(f"[{ticker}] Datos insuficientes.")
            
    actualizar_readme(resultados)
    
    # Guardar también en CSV para Streamlit
    if len(resultados) > 0:
        df_resultados = pd.DataFrame(resultados)
        df_resultados.to_csv("predicciones.csv", index=False)
        print("✅ predicciones.csv generado para Streamlit.")

if __name__ == "__main__":
    main()
