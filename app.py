# app.py - Sistema Predictivo de Calidad de Café Arábica (Versión FINAL con todos los Gráficos)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --- 0. Definiciones y Carga de Archivos ---
FILE_PATH = "df_arabica_clean.csv"
TARGET_COLUMN = 'Overall' 

# Cargar el Modelo, Encoders, y la Importancia 
try:
    model = joblib.load('coffee_quality_predictor.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_importance = joblib.load('feature_importance.pkl')
except FileNotFoundError:
    st.error("🛑 ERROR: Asegúrate de que los archivos '.pkl' estén en la misma carpeta.")
    st.stop() 

# Definiciones de los rangos numéricos
NUMERIC_RANGES = {
    'Aroma': (7.0, 9.0, 8.0), 'Flavor': (7.0, 9.0, 8.0),
    'Aftertaste': (7.0, 9.0, 8.0), 'Acidity': (7.0, 9.0, 8.0),
    'Body': (7.0, 9.0, 8.0), 'Balance': (7.0, 9.0, 8.0),
    'Uniformity': (9.0, 10.0, 10.0), 'Clean Cup': (9.0, 10.0, 10.0),
    'Sweetness': (9.0, 10.0, 10.0),
    'Altitude': (500, 3000, 1500),
    'Moisture Percentage': (0.0, 0.20, 0.12)
}

CATEGORICAL_OPTIONS = {
    col: list(le.classes_) for col, le in label_encoders.items()
}

# --- 1. Configuración de la Aplicación Streamlit ---
st.set_page_config(page_title="Sistema Predictivo de Café", layout="wide")
st.title("☕ Sistema Predictivo de Calidad de Café Arábica")
st.markdown("---")

tab_predict, tab_viz = st.tabs(["🔮 Predicción de Puntaje", "📈 Análisis de Dataset"])

# --- 2. BARRA LATERAL PARA ENTRADA DE DATOS ---
st.sidebar.header("📝 Ingreso de Características del Lote")
input_data = {}

# 2.1. Datos de Origen y Proceso (Expanders)
with st.sidebar.expander("📍 Datos de Origen y Proceso"):
    for feature, options in CATEGORICAL_OPTIONS.items():
        input_data[feature] = st.selectbox(f'{feature}', options, index=0, key=f'sb_{feature}')
    
    alt_min, alt_max, alt_def = NUMERIC_RANGES['Altitude']
    input_data['Altitude'] = st.slider('Altitude (metros)', min_value=alt_min, max_value=alt_max, value=alt_def, step=10, key='sl_alt')
    moist_min, moist_max, moist_def = NUMERIC_RANGES['Moisture Percentage']
    input_data['Moisture Percentage'] = st.slider('Moisture Percentage (%)', min_value=moist_min * 100, max_value=moist_max * 100, value=moist_def * 100, step=0.1, format="%.1f%%", key='sl_moist') / 100 

# 2.2. Puntajes de Cata (Columnas)
st.sidebar.subheader("🌟 Puntajes de Cata")

col1, col2 = st.sidebar.columns(2)
cata_features_to_input = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Uniformity', 'Clean Cup', 'Sweetness'] 

for i, feature in enumerate(cata_features_to_input):
    min_val, max_val, default_val = NUMERIC_RANGES[feature]
    col = col1 if i < 5 else col2 

    input_data[feature] = col.slider(
        f'{feature}',
        min_value=min_val,
        max_value=max_val,
        value=default_val,
        step=0.01,
        key=f'sl_{feature}'
    )

st.sidebar.markdown("---")
predict_button = st.sidebar.button('Calcular Puntaje Predicho')

# --- 3. Lógica de Predicción (Pestaña 1) ---

with tab_predict:
    
    # 1. VALIDACIÓN SIMPLE DE ALTITUD
    if input_data['Altitude'] < 500:
        st.warning("⚠️ Altitud baja: Un café Arábica de especialidad se suele cultivar por encima de los 500 metros. El resultado puede no ser fiable.")

    if predict_button:
        
        # 2. PROCESAMIENTO DE LA ENTRADA
        input_df = pd.DataFrame([input_data])
        
        for col, le in label_encoders.items():
            try:
                value_to_encode = input_df[col].iloc[0]
                input_df[col] = le.transform([value_to_encode])[0] 
            except ValueError:
                st.warning(f"Categoría '{value_to_encode}' para {col} no reconocida. Usando valor 0.")
                input_df[col] = 0

        feature_order = [
            'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Uniformity', 
            'Clean Cup', 'Sweetness', 
            'Altitude', 
            'Country of Origin', 
            'Variety', 
            'Processing Method', 
            'Moisture Percentage'
        ]
        
        try:
            input_processed = input_df.reindex(columns=feature_order, fill_value=0)
        except Exception as e:
            st.error(f"Error al reordenar columnas. Asegúrate de que el modelo fue entrenado. Error: {e}")
            st.stop() 

        # Realizar la Predicción
        prediction = model.predict(input_processed)[0]

        # 3. MOSTRAR EL RESULTADO CON FORMATO CONDICIONAL
        st.header("✨ Resultado de la Predicción")
        
        if prediction >= 85.0:
            score_style = "background-color: #4CAF50; color: white; padding: 20px; border-radius: 10px; font-size: 2.5em; text-align: center;" 
        elif prediction >= 80.0:
            score_style = "background-color: #FFC107; color: black; padding: 20px; border-radius: 10px; font-size: 2.5em; text-align: center;" 
        else:
            score_style = "background-color: #F44336; color: white; padding: 20px; border-radius: 10px; font-size: 2.5em; text-align: center;" 

        st.markdown(
            f'<div style="{score_style}">Puntaje de Calidad Predicho: **{prediction:.2f}** / 100</div>',
            unsafe_allow_html=True
        )
        
        st.markdown("---")

        # 4. RESUMEN DETALLADO DE LAS ENTRADAS
        st.subheader("📋 Resumen de las Características Ingresadas")
        
        input_summary = pd.DataFrame.from_dict(input_data, orient='index', columns=['Valor Ingresado'])
        
        if 'Overall' in input_summary.index:
            input_summary = input_summary.drop(index='Overall') 
            
        input_summary.loc['Moisture Percentage', 'Valor Ingresado'] = f"{input_data['Moisture Percentage'] * 100:.1f}%"
        
        st.dataframe(input_summary)

        st.markdown("---")
        
        # 5. GRÁFICO DE IMPORTANCIA DE CARACTERÍSTICAS
        st.subheader("💡 Contribución del Modelo a la Predicción")
        st.info("Este gráfico muestra qué características influyeron más en la predicción del puntaje.")

        fig_importance = px.bar(
            feature_importance, 
            orientation='h',
            title='Importancia Relativa de las Características',
            labels={'value': 'Importancia (Peso)', 'index': 'Característica'},
            color=feature_importance.values,
            color_continuous_scale=px.colors.sequential.Teal
        )
        fig_importance.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
        
        st.plotly_chart(fig_importance, use_container_width=True)

        st.markdown("---")
        
        # 6. GRÁFICO DE DISPERSIÓN PARA COMPARACIÓN (¡NUEVA MEJORA!)
        st.subheader("📍 Comparación de Altitud vs. Puntaje")
        st.info("Compara tu predicción (punto grande y rojo) con todos los cafés históricos del dataset.")

        # 6.1. Cargar el dataset para la visualización
        df_viz = pd.read_csv(FILE_PATH)
        df_viz['Altitude'] = pd.to_numeric(df_viz['Altitude'], errors='coerce')
        df_viz.dropna(subset=[TARGET_COLUMN, 'Altitude'], inplace=True)
        
        # 6.2. Crear el punto de la PREDICCIÓN
        predicted_point = pd.DataFrame({
            TARGET_COLUMN: [prediction],
            'Altitude': [input_data['Altitude']],
            'Tipo': ['Tu Predicción'],
            'Color': ['#FF0000'] 
        })
        
        # 6.3. Gráfico con Plotly
        fig_scatter = px.scatter(
            df_viz, 
            x='Altitude', 
            y=TARGET_COLUMN, 
            opacity=0.5,
            title='Predicción vs. Datos Históricos',
            labels={'Altitude': 'Altitud (metros)', TARGET_COLUMN: 'Puntaje Total'},
            hover_data=['Country of Origin', 'Variety']
        )
        
        # 6.4. Añadir el punto de la predicción al gráfico
        fig_scatter.add_scatter(
            x=predicted_point['Altitude'],
            y=predicted_point[TARGET_COLUMN],
            mode='markers',
            marker=dict(size=15, color='#FF0000', symbol='star'),
            name='Tu Predicción'
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    else:
        st.info("Ajusta los parámetros en la barra lateral izquierda y haz clic en 'Calcular Puntaje Predicho'.")

# --- 4. Análisis de Dataset (Pestaña 2) ---

with tab_viz:
    st.header("Análisis del Dataset de Calidad de Café")
    try:
        # El código para la pestaña de análisis sigue aquí, pero re-utiliza la carga de df_viz
        df_viz = pd.read_csv(FILE_PATH)
        df_viz['Altitude'] = pd.to_numeric(df_viz['Altitude'], errors='coerce')
        df_viz.dropna(subset=[TARGET_COLUMN, 'Country of Origin', 'Altitude'], inplace=True)

        st.subheader("Distribución de Puntaje por País")
        fig_country = px.box(df_viz, x='Country of Origin', y=TARGET_COLUMN, title='Distribución de Puntajes por País de Origen', labels={TARGET_COLUMN: "Puntaje Total", "Country of Origin": "País"}, color='Country of Origin')
        st.plotly_chart(fig_country, use_container_width=True)

        st.subheader("Relación entre Altitud y Puntaje")
        # El gráfico de dispersión de la pestaña 2
        fig_altitude = px.scatter(df_viz, x='Altitude', y=TARGET_COLUMN, hover_data=['Country of Origin', 'Variety'], title='Puntaje Total vs. Altitud', labels={'Altitude': 'Altitud (metros)', TARGET_COLUMN: 'Puntaje Total'})
        st.plotly_chart(fig_altitude, use_container_width=True)

    except Exception as e:
        st.warning(f"No se pudo cargar o visualizar el dataset para el análisis. Asegúrate de que '{FILE_PATH}' existe. Error: {e}")