import __main__
from preprocessing import text_preprocessing_function
from train import vectorization_function, scaling_function

# Inyectamos las funciones en __main__ para que el pipeline encuentre las referencias
__main__.text_preprocessing_function = text_preprocessing_function

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from joblib import load, dump
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import warnings




warnings.filterwarnings("ignore")

# Intentar cargar el pipeline preentrenado (ahora __main__ tiene las funciones inyectadas)
try:
    pipeline = load('pipeline.joblib')
except Exception as e:
    print("No se pudo cargar el pipeline preentrenado:", e)
    pipeline = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de solicitud para predicción: se espera una lista de textos.
class PredictionRequest(BaseModel):
    Textos_espanol: list

# Modelo de solicitud para reentrenamiento: se esperan textos y sus etiquetas.
class RetrainingRequest(BaseModel):
    Textos_espanol: list
    sdg: list

@app.post("/predict")
async def predict(request: PredictionRequest):
    global pipeline
    if pipeline is None:
        return {"error": "El modelo no está cargado. Por favor, reentrene el modelo primero."}
    # Convertir la lista de textos a una Serie de pandas
    textos = pd.Series(request.Textos_espanol).astype(str)
    predicciones = pipeline.predict(textos)
    probabilidades = pipeline.predict_proba(textos)
    return {
        "predicciones": predicciones.tolist(),
        "probabilidades": probabilidades.tolist()
    }

@app.post("/retrain")
async def retrain(request: RetrainingRequest):
    global pipeline
    from train import train_model  # Importar la función de reentrenamiento
    pipeline, metrics = train_model(request.Textos_espanol, request.sdg)
    dump(pipeline, 'pipeline.joblib')
    return metrics

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
