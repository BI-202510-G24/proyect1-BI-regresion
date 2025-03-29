from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib

# Importa las funciones para actualizar el vectorizador, escalador y features
from preprocessing import set_vectorizer, set_scaler, set_selected_features

# Rutas de los archivos persistidos
MODEL_PATH = "pipeline.joblib"
VECTORIZER_PATH = "vectorizer.joblib"
SCALER_PATH = "scaler.joblib"
SELECTED_FEATURES_PATH = "selected_features.joblib"

# Cargar el pipeline entrenado
try:
    pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    raise Exception(f"No se pudo cargar el pipeline: {e}")

# Cargar el vectorizador, escalador y features ajustados, y asignarlos al módulo de preprocesamiento
try:
    fitted_vectorizer = joblib.load(VECTORIZER_PATH)
    fitted_scaler = joblib.load(SCALER_PATH)
    selected_features = joblib.load(SELECTED_FEATURES_PATH)
    set_vectorizer(fitted_vectorizer)
    set_scaler(fitted_scaler)
    set_selected_features(selected_features)
except Exception as e:
    raise Exception(f"No se pudieron cargar vectorizer/scaler/features: {e}")

# Modelo de entrada para el endpoint /predict
class PredictRequest(BaseModel):
    Textos_espanol: List[str]

# Modelo de respuesta para el endpoint /predict
class PredictionResponse(BaseModel):
    predictions: List[int]

app = FastAPI(title="API de Predicción de Fake News")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest):
    textos = request.Textos_espanol
    if not textos:
        raise HTTPException(status_code=400, detail="La lista de textos está vacía")
    
    # Convertir la lista de textos en un DataFrame
    datatest = pd.DataFrame({"message": textos})
    
    try:
        y_pred_test = pipeline.predict(datatest["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")
    
    # Devuelve las predicciones en forma de lista
    return {"predictions": y_pred_test.tolist()}

# Para ejecutar la API, usa:
# uvicorn main:app --reload
