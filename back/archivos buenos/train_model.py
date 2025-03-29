# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from joblib import dump
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

# Importar funciones y variables desde preprocessing.py
from preprocessing import (
    procesar,
    text_preprocessing_function,
    vectorization_function,
    scaling_function,
    set_vectorizer,
    set_scaler,
    set_selected_features,
    select_top_features
)

# 1. Cargar los datos
datatrain = pd.read_csv('content/fake_news_spanish_small.csv', sep=';', encoding='utf-8')

# 2. Preparar la columna "message"
datatrain["message"] = datatrain["Titulo"].astype(str) + " " + datatrain["Descripcion"].astype(str)
datatrain.drop(columns=["ID", "Titulo", "Descripcion", "Fecha"], inplace=True)

X = datatrain["message"]
y = datatrain["Label"]

# 3. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 4. Preprocesar manualmente X_train para entrenar vectorizador y escalador
X_train_proc = X_train.apply(procesar)  # Limpieza textual

# Instanciar un CountVectorizer con parámetros para limitar la dimensionalidad
bv = CountVectorizer(
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.9,
    max_features=2000  # Limita el número de features
)
# Ajustar el vectorizador y escalador
bv.fit(X_train_proc)
X_train_vec = bv.transform(X_train_proc).toarray()

# Convertir a DataFrame para etiquetar las columnas
X_train_df = pd.DataFrame(X_train_vec, columns=bv.get_feature_names_out())

# Aplicar SMOTE usando el DataFrame
from imblearn.over_sampling import SMOTE
smt = SMOTE()
X_smote, y_smote = smt.fit_resample(X_train_df, y_train)

# Seleccionar las 1000 palabras (n-gramas) más frecuentes (opcional)
palabras_mas_usadas_smote1 = X_smote.sum(axis=0).nlargest(1000).index
X_smote_filtered = X_smote[palabras_mas_usadas_smote1]

# Ajustar el escalador usando el DataFrame filtrado
scaler = StandardScaler()
scaler.fit(X_smote_filtered)

# Guardar el vectorizador, escalador y las features seleccionadas
dump(bv, "vectorizer.joblib")
dump(scaler, "scaler.joblib")
dump(palabras_mas_usadas_smote1, "selected_features.joblib")

# Entrenar el modelo (por ejemplo, con GradientBoosting)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100],
    'max_depth': [5]
}
gb = GradientBoostingClassifier(random_state=0)
grid = GridSearchCV(gb, param_grid, cv=2)
grid.fit(X_smote_filtered, y_smote)
best_clf = grid.best_estimator_
print("Mejor modelo:", best_clf)

# Asignar el vectorizador y escalador globalmente para usarlos en el pipeline
from preprocessing import set_vectorizer, set_scaler, set_selected_features
set_vectorizer(bv)
set_scaler(scaler)
set_selected_features(palabras_mas_usadas_smote1)

# Construir el pipeline final, incluyendo la etapa de selección de features
from sklearn.pipeline import Pipeline, FunctionTransformer
from preprocessing import text_preprocessing_function, vectorization_function, scaling_function, select_top_features

pipeline = Pipeline([
    ('text_preprocessing', FunctionTransformer(text_preprocessing_function)),
    ('vectorization', FunctionTransformer(vectorization_function)),
    ('feature_selection', FunctionTransformer(select_top_features)),
    ('scaling', FunctionTransformer(scaling_function)),
    ('classification', best_clf)
])

# Guardar el pipeline entrenado
dump(pipeline, 'pipeline.joblib')
print("Pipeline guardado exitosamente en pipeline.joblib")


# 11. (Opcional) Validar brevemente con X_test
X_test_proc = X_test.apply(procesar)
X_test_vec = bv.transform(X_test_proc).toarray()
# Convertir a DataFrame para aplicar la selección de columnas
X_test_df = pd.DataFrame(X_test_vec, columns=bv.get_feature_names_out())
X_test_selected = X_test_df[palabras_mas_usadas_smote1]
X_test_scaled = scaler.transform(X_test_selected)
y_pred = best_clf.predict(X_test_scaled)
print("Accuracy en test:", accuracy_score(y_test, y_pred))
