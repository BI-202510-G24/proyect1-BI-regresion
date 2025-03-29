import pandas as pd
import requests
import json
import sys

def leer_excel_y_construir_json(archivo_excel):
    # Leer el archivo Excel
    df = pd.read_excel(archivo_excel)
    
    # Asegurarse de que el archivo contiene las columnas necesarias
    if 'message' not in df.columns or 'Label' not in df.columns:
        print("El archivo no contiene las columnas necesarias ('message' y 'Label').")
        sys.exit(1)
    
    # Convertir los datos a listas
    textos = df['message'].tolist()
    etiquetas = df['Label'].tolist()

    # Construir el JSON para el reentrenamiento
    json_data = {
        "message": textos,
        "Label": etiquetas
    }
    
    return json_data

def enviar_reentrenamiento(json_data, url):
    # Configurar la URL del endpoint del backend
    endpoint = f"{url}/retrain"
    
    # Enviar la solicitud POST
    headers = {'Content-Type': 'application/json'}
    response = requests.post(endpoint, data=json.dumps(json_data), headers=headers)

    # Imprimir el resultado de la solicitud
    if response.status_code == 200:
        print("Reentrenamiento exitoso.")
        print("Respuesta del servidor:", response.json())
    else:
        print(f"Error en la solicitud: {response.status_code}")
        print("Detalles:", response.text)

if __name__ == "__main__":
    # Comprobar si el archivo Excel fue pasado como argumento
    if len(sys.argv) != 3:
        print("Uso: python try_api.py <ruta_al_archivo_excel> <url_backend>")
        sys.exit(1)

    # Obtener el archivo y la URL del backend desde los argumentos
    archivo_excel = sys.argv[1]
    url_backend = sys.argv[2]

    # Leer el archivo Excel y construir el JSON
    json_data = leer_excel_y_construir_json(archivo_excel)

    # Enviar el JSON al backend
    enviar_reentrenamiento(json_data, url_backend)
