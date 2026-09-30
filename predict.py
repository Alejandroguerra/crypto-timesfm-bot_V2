import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import timesfm # Librería oficial de Google

# Configuración
TICKERS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", 
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "NEARUSDT"
]
BINANCE_URL = "https://api.binance.com/api/v3/klines"
CONTEXT_LEN = 512 # Ventana histórica que mirará el modelo (512 horas)
HORIZON_LEN = 12  # Horas hacia el futuro que va a predecir

def inicializar_modelo_timesfm():
    """
    Descarga e inicializa el modelo pre-entrenado de Google.
    Se configura para usar CPU, ideal para GitHub Actions.
    """
    print("Cargando modelo TimesFM desde Hugging Face (esto puede tardar unos minutos)...")
    tfm = timesfm.TimesFm(
        context_len=CONTEXT_LEN,
        horizon_len=HORIZON_LEN,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend="cpu" # Forzamos CPU para que corra gratis en GitHub Actions
    )
    # Descarga los pesos del modelo oficial
    tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
    print("✅ Modelo cargado correctamente.")
    return tfm

def obtener_datos_binance(symbol, interval="1h", limit=CONTEXT_LEN):
    """Descarga el historial de precios desde Binance."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        response = requests.get(BINANCE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume", 
            "close_time", "quote_asset_volume", "number_of_trades", 
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df['close'] = df['close'].astype(float)
        return df
    except Exception as e:
        print(f"Error descargando {symbol}: {e}")
        return None

def predecir_con_timesfm(tfm, df):
    """
    Utiliza TimesFM para proyectar el precio.
    """
    historia_precios = df['close'].values
    precio_actual = historia_precios[-1]

    # Inferencia con TimesFM (freq=[0] indica alta frecuencia)
    forecast_result = tfm.forecast(inputs=[historia_precios], freq=[0])
    
    # Extraemos la predicción puntual
    predicciones_futuras = forecast_result[0][0] 
    precio_proyectado_12h = predicciones_futuras[-1] 
    
    # Calcular la variación esperada
    variacion_pct = ((precio_proyectado_12h - precio_actual) / precio_actual) * 100
    
    # Generar Señal
    if variacion_pct > 1.5:
        senal = "COMPRA 🟢"
    elif variacion_pct < -1.5:
        senal = "VENTA 🔴"
    else:
        senal = "MANTENER ⚪"
        
    return senal, precio_actual, variacion_pct, precio_proyectado_12h

def actualizar_readme(resultados):
    """Reescribe el archivo README.md con la nueva tabla"""
    fecha_actual = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    md = f"# 🧠 TimesFM Crypto Predictor\n\n"
    md += f"**Última actualización:** `{fecha_actual}`\n\n"
    md += "> Predicciones generadas utilizando el modelo fundacional *TimesFM* de Google Research. Horizonte de predicción: 12 horas.\n\n"
    
    md += "| Criptomoneda | Precio Actual | Proyección (12h) | Variación Est. | Señal Sugerida |\n"
    md += "| :--- | :--- | :--- | :--- | :--- |\n"
    
    for res in resultados:
        nombre = res['ticker'].replace('USDT', '')
        precio = f"${res['precio']:.4f}".rstrip('0').rstrip('.')
        proy = f"${res['proyectado']:.4f}".rstrip('0').rstrip('.')
        var = f"{res['variacion']:.2f}%"
        
        if res['variacion'] > 0: var = f"+{var} 📈"
        elif res['variacion'] < 0: var = f"{var} 📉"
            
        md += f"| **{nombre}** | {precio} | {proy} | {var} | **{res['senal']}** |\n"

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("✅ README.md actualizado.")

def main():
    tfm = inicializar_modelo_timesfm()
    resultados = []
    
    for ticker in TICKERS:
        df = obtener_datos_binance(ticker, interval="1h", limit=CONTEXT_LEN)
        
        if df is not None and len(df) == CONTEXT_LEN:
            senal, precio, variacion, proyectado = predecir_con_timesfm(tfm, df)
            resultados.append({
                "ticker": ticker,
                "precio": precio,
                "proyectado": proyectado,
                "variacion": variacion,
                "senal": senal
            })
            print(f"{ticker}: Procesado ({senal})")
        else:
            print(f"[{ticker}] Datos insuficientes (necesarios {CONTEXT_LEN}).")
            
    actualizar_readme(resultados)

if __name__ == "__main__":
    main()
