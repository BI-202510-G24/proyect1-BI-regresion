import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from preprocessing import text_preprocessing_function
import re
import unicodedata
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

def normalizar_palabra(palabra):
    nfkd_form = unicodedata.normalize('NFKD', palabra)
    palabra_sin_diacriticos = ''.join([c for c in nfkd_form if not unicodedata.combining(c)])
    return re.sub(r'[^a-zA-Z]', '', palabra_sin_diacriticos).lower()

def get_most_frequent_words(text_series, n):
    all_words = []
    for text in text_series:
        for word in text.split():
            if word not in set(stopwords.words('spanish')):
                all_words.append(normalizar_palabra(word))
    word_counts = Counter(all_words)
    most_common = word_counts.most_common(n)
    return most_common

# Transformer personalizado para vectorizar
class CustomVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, ngram_range=(1,3), min_df=2, max_df=0.9, max_features=2000):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.vectorizer = CountVectorizer(
            ngram_range=self.ngram_range, 
            min_df=self.min_df, 
            max_df=self.max_df, 
            max_features=self.max_features
        )

    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        transformed = self.vectorizer.transform(X)
        return pd.DataFrame(
            transformed.toarray(), 
            columns=self.vectorizer.get_feature_names_out()
        )

# Transformer personalizado para escalar
class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        transformed = self.scaler.transform(X)
        return pd.DataFrame(transformed, columns=X.columns)

def train_model(X, y):
    """
    X: lista o Serie de textos.
    y: lista o Serie de etiquetas.
    """
    # Cargar datos iniciales desde CSV (usando separador ;)
    try:
        df = pd.read_csv('../content/fake_news_spanish.csv', sep=';')
        df["message"] = (df["Titulo"] + " " + df["Descripcion"]).astype(str)
        df = df.drop(columns=["ID", "Titulo", "Descripcion", "Fecha"])
    except Exception as e:
        print("Error al cargar el CSV inicial:", e)
        df = pd.DataFrame(columns=["message", "Label"])
        
    new_data = pd.DataFrame({'message': X, 'Label': y})
    df = pd.concat([df, new_data], ignore_index=True)
    # Guardar el CSV actualizado
    df.to_csv('../content/fake_news_spanish.csv', sep=';', index=False)
    
    X_combined = df['message']
    y_combined = df['Label']
    
    pipeline = Pipeline([
        ('text_preprocessing', FunctionTransformer(text_preprocessing_function)),
        ('vectorization', CustomVectorizer()),
        ('scaling', CustomScaler()),
        ('classification', GradientBoostingClassifier(n_estimators=300, max_depth=5, random_state=1234))
    ])
    
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=1234
    )
    pipeline.fit(X_train, y_train)
    y_pred_validation = pipeline.predict(X_validation)
    
    metrics = {
        'accuracy': accuracy_score(y_validation, y_pred_validation),
        'recall': recall_score(y_validation, y_pred_validation, average='macro'),
        'precision': precision_score(y_validation, y_pred_validation, average='macro'),
        'f1score': f1_score(y_validation, y_pred_validation, average='macro'),
        'words': get_most_frequent_words(df['message'], n=10)
    }
    
    return pipeline, metrics
