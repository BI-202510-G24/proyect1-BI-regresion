from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
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
    messages: List[str]

# Modelo de respuesta para el endpoint /predict
class PredictionResponse(BaseModel):
    predictions: List[int]
    
    
# Modelo de entrada para el endpoint /retrain
class RetrainRequest(BaseModel):
    messages: List[str]
    sdg: List[int]


app = FastAPI(title="API de Predicción de Fake News")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest):
    textos = request.messages
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


# Importar la función de reentrenamiento desde retrain_model.py
from retrain_model import retrain_model_api

@app.post("/retrain")
def retrain(request: RetrainRequest):
    textos = request.messages
    etiquetas = request.sdg
    if not textos or not etiquetas:
        raise HTTPException(status_code=400, detail="Se deben proporcionar textos y etiquetas")
    
    try:
        metrics = retrain_model_api(textos, etiquetas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante el reentrenamiento: {e}")
    
    return metrics

# Para ejecutar la API, usa:
# uvicorn main:app --reload
