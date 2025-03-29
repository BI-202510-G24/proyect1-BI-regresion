# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

MODEL_PATH = "pipeline.joblib"

# Cargar el pipeline
try:
    pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    raise Exception(f"No se pudo cargar el modelo: {e}")

# Modelos de datos para la API
class PredictionInstance(BaseModel):
    message: str

class PredictionResponse(BaseModel):
    prediction: int
    probability: float

class RetrainInstance(BaseModel):
    message: str
    Label: int

class RetrainRequest(BaseModel):
    data: List[RetrainInstance]

app = FastAPI(title="API de Clasificación y Reentrenamiento de Modelo")

@app.post("/predict", response_model=List[PredictionResponse])
def predict(instances: List[PredictionInstance]):
    if not instances:
        raise HTTPException(status_code=400, detail="No se proporcionaron datos.")
    
    # Convertir instancias a DataFrame
    df = pd.DataFrame([instance.dict() for instance in instances])
    try:
        preds = pipeline.predict(df["message"])
        probs = pipeline.predict_proba(df["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")

    response = []
    for i, pred in enumerate(preds):
        try:
            class_index = list(pipeline.classes_).index(pred)
            prob = probs[i][class_index]
        except Exception:
            prob = None
        response.append(PredictionResponse(prediction=int(pred), probability=float(prob)))
    return response

@app.post("/retrain")
def retrain(request: RetrainRequest):
    if not request.data:
        raise HTTPException(status_code=400, detail="No se proporcionaron datos para reentrenar.")
    
    # Convertir datos a DataFrame
    df = pd.DataFrame([instance.dict() for instance in request.data])
    X = df["message"]
    y = df["Label"]
    
    # Dividir para evaluar
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)
    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante el reentrenamiento: {e}")
    
    try:
        y_pred = pipeline.predict(X_val)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al calcular las métricas: {e}")
    
    try:
        joblib.dump(pipeline, MODEL_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar el modelo: {e}")
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mensaje": "Modelo reentrenado exitosamente utilizando reentrenamiento total."
    }

# Para ejecutar la API, en la terminal ejecutar:
# uvicorn main:app --reload
