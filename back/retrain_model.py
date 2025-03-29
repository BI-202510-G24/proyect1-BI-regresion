# retrain_model.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.metrics import precision_score, recall_score, f1_score

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

def retrain_model_api(texts, labels):
    """
    Función para reentrenar el modelo con nuevos datos recibidos vía API.
    Se espera que 'texts' sea una lista de strings (cada uno con título y descripción)
    y 'labels' una lista de enteros (la variable objetivo).
    Retorna un diccionario con las métricas de desempeño (precision, recall y f1_score).
    """
    # Crear DataFrame a partir de los datos recibidos
    df = pd.DataFrame({"message": texts, "sdg": labels})
    
    # Si hay más de un registro, realizar train-test split; de lo contrario se entrena y evalúa sobre el mismo registro
    if len(df) > 1:
        X_train, X_test, y_train, y_test = train_test_split(df["message"], df["sdg"], test_size=0.3, random_state=0)
    else:
        X_train, X_test, y_train, y_test = df["message"], df["message"], df["sdg"], df["sdg"]

    # Preprocesamiento de textos (los textos ya contienen título y descripción)
    X_train_proc = X_train.apply(procesar)
    
    # Instanciar y ajustar CountVectorizer con parámetros consistentes
    bv = CountVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.9, max_features=2000)
    bv.fit(X_train_proc)
    X_train_vec = bv.transform(X_train_proc).toarray()
    X_train_df = pd.DataFrame(X_train_vec, columns=bv.get_feature_names_out())

    # Aplicar SMOTE para balancear las clases (si es posible)
    smt = SMOTE()
    try:
        X_smote, y_smote = smt.fit_resample(X_train_df, y_train)
    except Exception as e:
        X_smote, y_smote = X_train_df, y_train

    # Seleccionar las 1000 palabras (n-gramas) más frecuentes
    selected_features = X_smote.sum(axis=0).nlargest(1000).index
    X_smote_filtered = X_smote[selected_features]

    # Ajustar el escalador con los datos filtrados
    scaler = StandardScaler()
    scaler.fit(X_smote_filtered)

    # Entrenar el modelo usando GradientBoostingClassifier a través de GridSearchCV
    param_grid = {'n_estimators': [100], 'max_depth': [5]}
    gb = GradientBoostingClassifier(random_state=0)
    grid = GridSearchCV(gb, param_grid, cv=2)
    grid.fit(X_smote_filtered, y_smote)
    best_clf = grid.best_estimator_

    # Actualizar las variables globales en preprocessing para el pipeline
    set_vectorizer(bv)
    set_scaler(scaler)
    set_selected_features(selected_features)

    # Construir el pipeline final con las mismas etapas que en el entrenamiento original
    pipeline_new = Pipeline([
        ('text_preprocessing', FunctionTransformer(text_preprocessing_function)),
        ('vectorization', FunctionTransformer(vectorization_function)),
        ('feature_selection', FunctionTransformer(select_top_features)),
        ('scaling', FunctionTransformer(scaling_function)),
        ('classification', best_clf)
    ])

    # Guardar los componentes actualizados (reemplazando los anteriores)
    dump(pipeline_new, 'pipeline.joblib')
    dump(bv, 'vectorizer.joblib')
    dump(scaler, 'scaler.joblib')
    dump(selected_features, 'selected_features.joblib')

    # Evaluación: Preprocesar y transformar el set de prueba
    X_test_proc = X_test.apply(procesar)
    X_test_vec = bv.transform(X_test_proc).toarray()
    X_test_df = pd.DataFrame(X_test_vec, columns=bv.get_feature_names_out())
    X_test_selected = X_test_df.reindex(columns=selected_features, fill_value=0)
    X_test_scaled = scaler.transform(X_test_selected)
    y_pred = best_clf.predict(X_test_scaled)

    # Calcular las métricas de desempeño
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
