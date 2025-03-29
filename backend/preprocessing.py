import re
import unicodedata
import nltk
import contractions
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer

# Descargas necesarias de NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def a_minusculas(palabras):
    return [palabra.lower() for palabra in palabras]

def eliminar_numeros(palabras):
    return [re.sub(r'\d+', '', palabra) for palabra in palabras]

def eliminar_puntuacion(palabras):
    import string
    return [palabra for palabra in palabras if palabra not in string.punctuation]

def remover_no_ascii(palabras):
    return [palabra for palabra in palabras if palabra.isascii()]

def eliminar_stopwords(palabras):
    sw = set(stopwords.words('spanish'))
    return [palabra for palabra in palabras if palabra not in sw]

def aplicar_stemming_y_lemmatizacion(palabras):
    stemmer = SnowballStemmer('spanish')
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(stemmer.stem(palabra)) for palabra in palabras]

def preprocesar_palabras(palabras):
    palabras = a_minusculas(palabras)
    palabras = eliminar_numeros(palabras)
    palabras = eliminar_puntuacion(palabras)
    palabras = remover_no_ascii(palabras)
    palabras = eliminar_stopwords(palabras)
    return palabras

def procesar(texto):
    texto = contractions.fix(texto)
    palabras = word_tokenize(texto)
    palabras = preprocesar_palabras(palabras)
    palabras = aplicar_stemming_y_lemmatizacion(palabras)
    return " ".join(palabras)

def text_preprocessing_function(x):
    # Se espera que x sea una Serie de pandas
    return x.apply(procesar)
