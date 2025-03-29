# Machine Learning Backend

This is the backend for the machine learning model, implemented using FastAPI. It provides a RESTful API that serves predictions from a machine learning model, as well as a re-training endpoint to update the model.

## Installation

Follow these steps to install and set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/ddi4z/Proyecto-inteligencia-de-negocios.git
   cd Proyecto-inteligencia-de-negocios/backend
   ```
2. Install virtualenv if you do not have this Python tool
   ```bash
   pip install virtualenv
   ```
3. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
4. Install the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To start the FastAPI backend, run the following command:

```bash
uvicorn main:app --reload
```

The backend will be available at `http://localhost:8000`.

## SDG Classification Model

The backend uses a pre-trained machine learning model to classify Spanish texts into Sustainable Development Goals (SDGs). The model is trained on a dataset of Spanish texts labeled with SDG categories. The model is loaded when the backend starts and is used to make predictions on new texts. The meaning of each SDG category in the dataset is as follows:

- `3`: Good Health and Well-being
- `4`: Quality Education
- `5`: Gender Equality

## Endpoints

The backend exposes two endpoints: one for making predictions and another for retraining the model with new data. Below is a detailed explanation of how each endpoint works, including examples of request and response formats.

### 1. POST `/predict`

This endpoint accepts a list of Spanish texts and returns a prediction along with the associated probabilities for each text. The texts are processed through a pre-trained pipeline that includes text preprocessing, vectorization, and classification.

#### Request

- **Method**: `POST`
- **Endpoint**: `/predict`
- **Request Body**: JSON object containing a list of texts in Spanish (`Textos_espanol`) that are used as input for predictions.

Example request body:

```json
{
  "Textos_espanol": [
    "Este es el primer texto para predecir.",
    "Aquí otro ejemplo de entrada de texto."
  ]
}
```

#### Response

- **Status Code**: `200 OK`
- **Response Body**: JSON object containing two lists:
  - `predicciones`: A list of predicted classes for each input text.
  - `probabilidades`: A list of probabilities associated with each prediction.

Example response:

```json
{
  "predicciones": [3, 4],
  "probabilidades": [
    [0.85, 0.15, 0.0],
    [0.4, 0.6, 0.0]
  ]
}
```

In this example:

- `"predicciones"` contains the predicted class labels (e.g., `3`, `4` or `5`).
- `"probabilidades"` contains the probabilities for each class.

#### Example

Execute the following command to make a prediction using the `/predict` endpoint:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "Textos_espanol": [

    "La OMS también estableció en 2000 la Red mundial de respuesta ante brotes y alertas (GOARN) con el objetivo de gestionar brotes y alertas de enfermedades a nivel internacional. Sin embargo, algunos han cuestionado si el GOARN actual pone demasiado énfasis en evitar que los brotes epidémicos se propaguen a los países desarrollados en lugar de prevenir dichos brotes en los países en desarrollo en primer lugar (Davies, 2008, Elbe, 2010). El aumento de la esperanza de vida explica poco del aumento de los costos de atención de la salud experimentado en décadas anteriores",

    "La próxima década, por lo tanto, será en gran parte de crecimiento en la provisión y participación en la secundaria superior. Esto requerirá una mayor eficiencia en el despliegue de maestros y una mejor alineación de las ofertas de programas con el tamaño de la escuela. Una mayor participación en la escuela secundaria superior también requerirá una mayor atención a la relevancia de la educación para la vida, el trabajo y las perspectivas de aprendizaje de los estudiantes.",

    "Los primeros aÃ±os del auge liderado por las exportaciones de Taiwan fueron impulsados por la entrada de estas mujeres en las fÃ¡bricas de exportaciÃ³n. En la década de 1970, cuando Taiwán enfrentó escasez de mano de obra, un sistema de fábrica satélite patrocinado por el estado hizo que el trabajo industrial fuera más consistente con los roles femeninos tradicionales, lo que permitió aumentar la oferta laboral de esposas y madres (Hsiung 1996). Este tipo de flexibilidad estructural en relación con el trabajo de las mujeres y los roles familiares es una característica que persiste en los mercados laborales taiwaneses en la actualidad y encaja con las expectativas de que las mujeres contribuyan financieramente a sus familias (Yu 2009). Como consecuencia, los aumentos en la participación laboral femenina han sido grandes y sostenidos, pasando del 35,5 por ciento en 1970 al 44,5 por ciento en 1990 y al 50,2 por ciento en 2012.''7 En comparación con otras economías de Asia oriental, que suelen tener algunos de las brechas salariales de género más grandes del mundo, las mujeres en Taiwán experimentan una brecha salarial de género más cercana a la norma de los países liberales industrializados avanzados: la relación salarial de género para los ingresos mensuales promedio en la industria y los servicios fue de 81,2 en 2012. Consideremos si esta dinámica ha acercó a Taiwán al camino bajo/feminización de la responsabilidad o al camino alto/caso de igualdad de género."
  ]
}'
```

Which should predict the SDG categories: `[3, 4, 5]`.

### 2. POST `/retrain`

This endpoint accepts new training data, including both text inputs and their corresponding class labels, to retrain the machine learning model. It returns performance metrics after retraining, including precision, recall, and F1-score.

#### Request

- **Method**: `POST`
- **Endpoint**: `/retrain`
- **Request Body**: JSON object containing two lists:
  - `Textos_espanol`: A list of Spanish texts for retraining.
  - `sdg`: A list of labels (target values) corresponding to the input texts.

Example request body:

```json
{
  "Textos_espanol": [
    "Este es un nuevo texto de entrenamiento.",
    "Segundo texto para reentrenar el modelo."
  ],
  "sdg": [0, 1]
}
```

#### Response

- **Status Code**: `200 OK`
- **Response Body**: JSON object containing performance metrics after retraining:
  - `accuracy`: Accuracy score for the updated model.
  - `precision`: Precision score for the updated model.
  - `recall`: Recall score for the updated model.
  - `f1_score`: F1-score for the updated model.

Example response:

```json
{
  "accuracy": 0.91,
  "precision": 0.92,
  "recall": 0.89,
  "f1_score": 0.9
}
```

In this example, the response indicates the performance of the model on the retraining data.

#### Example

You can use the following script to automatically transform an Excel file into a JSON object and send it to the `/retrain` endpoint:

```bash
python backend/scripts/try_api.py content/ODScat_345.xlsx http://localhost:8000/
```

Which should return metrics similar to the following:

```json
{
  "accuracy": 0.9691358024691358,
  "precision": 0.9689786312527277,
  "recall": 0.9694981095292956,
  "f1_score": 0.9692214959916104
}
```
