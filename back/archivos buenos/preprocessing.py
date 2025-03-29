# preprocessing.py

import re
import string
import contractions
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

# Descargas de recursos de nltk (solo la primera vez que ejecutes)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Variables globales para el vectorizador y el escalador
bv = CountVectorizer()
scaler = StandardScaler()

def set_vectorizer(vectorizer):
    global bv
    bv = vectorizer

def set_scaler(s):
    global scaler
    scaler = s

# ----- Funciones para la selección de features -----
selected_features = None

def set_selected_features(features):
    """
    Almacena las features seleccionadas para usarlas en el pipeline.
    """
    global selected_features
    selected_features = features


def select_top_features(df):
    """
    Selecciona las columnas definidas en 'selected_features'. Si alguna columna no está presente
    en el DataFrame de entrada, se rellena con 0, garantizando que siempre se devuelva un DataFrame
    con las mismas columnas que se usaron durante el entrenamiento.
    """
    if selected_features is not None:
        # Reindexar para asegurarse de que se tengan todas las columnas de entrenamiento,
        # rellenando con 0 las que falten.
        return df.reindex(columns=selected_features, fill_value=0)
    else:
        return df


# ----- Funciones de preprocesamiento de texto -----
def aMinusculas(palabras):
    return [palabra.lower() for palabra in palabras]

def eliminarNumeros(palabras):
    return [re.sub(r'\d+', '', palabra) for palabra in palabras]

def eliminarPuntuacion(palabras):
    return [palabra for palabra in palabras if palabra not in string.punctuation]

def removerNoAscii(palabras):
    return [palabra for palabra in palabras if palabra.isascii()]

def eliminarStopwords(palabras):
    sw = set(stopwords.words('spanish'))
    return [palabra for palabra in palabras if palabra not in sw]

def aplicarStemmingYLematizacion(palabras):
    stemmer = SnowballStemmer('spanish')
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(stemmer.stem(palabra)) for palabra in palabras]

def preprocesar(palabras):
    palabras = aMinusculas(palabras)
    palabras = eliminarNumeros(palabras)
    palabras = eliminarPuntuacion(palabras)
    palabras = removerNoAscii(palabras)
    palabras = eliminarStopwords(palabras)
    return palabras

def procesar(texto):
    texto = contractions.fix(texto)
    palabras = word_tokenize(texto)
    palabras = preprocesar(palabras)
    palabras = aplicarStemmingYLematizacion(palabras)
    return " ".join(palabras)

def text_preprocessing_function(x):
    return x.apply(procesar)

def vectorization_function(x):
    return pd.DataFrame(bv.transform(x).toarray(), columns=bv.get_feature_names_out())

def scaling_function(x):
    """
    Escala las features utilizando el escalador global y devuelve un DataFrame
    con las columnas definidas en selected_features, garantizando que el orden
    y el conjunto de columnas coincidan con el ajuste realizado durante el entrenamiento.
    """
    # Forzamos que x tenga exactamente las columnas usadas en el entrenamiento
    df = x.reindex(columns=selected_features, fill_value=0)
    scaled = scaler.transform(df)
    return pd.DataFrame(scaled, columns=selected_features)