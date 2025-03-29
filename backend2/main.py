from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from joblib import load, dump
import uvicorn
from train import train_model
import warnings
from fastapi.middleware.cors import CORSMiddleware
warnings.filterwarnings("ignore")

# Cargar el pipeline previamente entrenado
pipeline = load('../modelo_clasificacion.joblib')

# Inicializar la aplicación FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Modelo para la solicitud de predicción
class PredictionRequest(BaseModel):
    ID: list
    Titulo: list
    Descripcion: list
    Fecha: list


# Modelo para la solicitud de reentrenamiento
class RetrainingRequest(BaseModel):
    ID: list
    Label: list
    Titulo: list
    Descripcion: list
    Fecha: list


# Endpoint 1: Predicción
@app.post("/predict")
async def predict(request: PredictionRequest):

    # Extraer los datos de la solicitud
    titulo = pd.Series(request.Titulo).astype(str)
    descripccion = pd.Series(request.Descripcion).astype(str)

    textos = (titulo + " " + descripccion).astype(str)
    # Usar el pipeline para hacer predicciones
    predicciones = pipeline.predict(textos)
    probabilidades = pipeline.predict_proba(textos)
    
    # Devolver las predicciones y probabilidades
    return {"predicciones": predicciones.tolist(), "probabilidades": probabilidades.tolist()}

# Endpoint 2: Reentrenamiento del modelo
@app.post("/retrain")
async def retrain(request: RetrainingRequest):

    # Extraer los datos de la solicitud
    titulo = pd.Series(request.Titulo).astype(str)
    descripccion = pd.Series(request.Descripcion).astype(str)
    textos = (titulo + " " + descripccion).astype(str)

    etiquetas = pd.Series(request.Label).astype(int)
    
    # Entrenar un nuevo modelo con los datos proporcionados
    pipeline, metrics = train_model(textos, etiquetas)
    
    # Guardar el modelo actualizado
    dump(pipeline, 'pipeline.joblib')
    
    return metrics

# Para correr la aplicación
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)