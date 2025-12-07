# model_training.py - VERSIÓN FINAL Y ROBUSTA (Añade Guardado de Importancia)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib 
import re 

# --- 1. Definiciones y Carga ---
FILE_PATH = "df_arabica_clean.csv"
TARGET_COLUMN = 'Overall' 

FEATURES = [
    'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Uniformity', 
    'Clean Cup', 'Sweetness', 
    'Altitude', 
    'Country of Origin', 
    'Variety', 
    'Processing Method', 
    'Moisture Percentage'
]
categorical_cols = ['Country of Origin', 'Variety', 'Processing Method']
numeric_cols = [f for f in FEATURES if f not in categorical_cols]


try:
    df = pd.read_csv(FILE_PATH)
    print(f"✅ Dataset '{FILE_PATH}' cargado exitosamente. Filas iniciales: {len(df)}")
except FileNotFoundError:
    print(f"🛑 ERROR: Archivo '{FILE_PATH}' no encontrado.")
    exit()

required_cols = FEATURES + [TARGET_COLUMN]
for col in required_cols:
    if col not in df.columns:
        print(f"🛑 ERROR: La columna requerida '{col}' no existe en el dataset. Revisa la lista FEATURES.")
        exit()

df_model = df[required_cols].copy()
df_model[TARGET_COLUMN] = pd.to_numeric(df_model[TARGET_COLUMN], errors='coerce')

# --- 2. Limpieza de Datos y Conversión (Imputación Total) ---

def clean_altitude(alt):
    if pd.isna(alt): return np.nan
    try: return float(alt)
    except:
        alt = str(alt).lower().replace(',', '').strip()
        if '-' in alt:
            parts = alt.split('-');
            if len(parts) == 2:
                try: return (float(parts[0]) + float(parts[1])) / 2
                except: pass
        match = re.search(r'(\d+)', alt);
        if match: return float(match.group(1));
        return np.nan 

df_model.loc[:, 'Altitude'] = df_model['Altitude'].apply(clean_altitude)
df_model.dropna(subset=[TARGET_COLUMN], inplace=True) 

for col in categorical_cols:
    df_model.loc[:, col] = df_model[col].fillna('N/A_Missing').astype(str)

for col in numeric_cols:
    if df_model[col].isnull().any():
        try:
            mean_value = df_model[col].mean()
            df_model.loc[:, col] = df_model[col].fillna(mean_value)
        except TypeError:
            df_model.loc[:, col] = df_model[col].fillna(0) 

# --- 3. Pre-procesamiento de Características Categóricas ---

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model.loc[:, col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le

joblib.dump(label_encoders, 'label_encoders.pkl')

# --- 4. Entrenamiento del Modelo (Scikit-learn) ---

X = df_model.drop(columns=[TARGET_COLUMN])
y = df_model[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"📊 R2 Score del modelo en datos de prueba: {score:.4f}")

# --- 5. Guardar el Modelo Entrenado y la Importancia (¡CAMBIOS AQUÍ!) ---

# 5.1. Guardar la importancia de las características
feature_names = X.columns.tolist()
feature_importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

IMPORTANCE_FILENAME = 'feature_importance.pkl'
joblib.dump(feature_importance, IMPORTANCE_FILENAME)
print(f"✅ Importancia de características guardada en '{IMPORTANCE_FILENAME}'.")

# 5.2. Guardar el Modelo principal
MODEL_FILENAME = 'coffee_quality_predictor.pkl'
joblib.dump(model, MODEL_FILENAME)
print(f"✅ Modelo guardado en '{MODEL_FILENAME}'.")