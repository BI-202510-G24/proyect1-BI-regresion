# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from joblib import dump
from preprocessing import text_preprocessing_function, vectorization_function, scaling_function, set_vectorizer, set_scaler

# Cargar los datos
datatrain = pd.read_csv('content/fake_news_spanish.csv', sep=';', encoding='utf-8')
datatest = pd.read_csv('content/fake_news_test.csv', sep=';', encoding='utf-8')

# Preparar el conjunto de entrenamiento
textos = datatrain.copy()
textos["message"] = (textos["Titulo"] + " " + textos["Descripcion"]).astype(str)
textos = textos.drop(columns=["ID", "Titulo", "Descripcion", "Fecha"])

# En este ejemplo usamos directamente la columna "message" para vectorizar;
# en otros casos se podría aplicar un preprocesamiento previo y guardar en "words".
textos["words"] = textos["message"]

# Inicializar el vectorizador
bv = CountVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.9, max_features=2000)
bv_matrix_message = bv.fit_transform(textos["words"])
bv_lista_message = bv_matrix_message.toarray()

vocab_messages = bv.get_feature_names_out()
bv_df = pd.DataFrame(bv_lista_message, columns=vocab_messages)
bv_df = pd.concat([textos.drop(columns=["message", "words"]), bv_df], axis=1)

# Variable objetivo y características
Y = bv_df['Label']
X = bv_df.drop(['Label'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Balanceo con SMOTE
smt = SMOTE()
X_smote, Y_smote = smt.fit_resample(X_train, Y_train)
Y_smote = Y_smote.values.ravel()

# Escalamiento
scaler = StandardScaler()
X_smote = pd.DataFrame(scaler.fit_transform(X_smote), columns=vocab_messages)
X_test = pd.DataFrame(scaler.transform(X_test), columns=vocab_messages)

# (Opcional) Seleccionar las 1000 palabras más frecuentes
palabras_mas_usadas_smote1 = X_smote.sum(axis=0).nlargest(1000).index
X_smote1 = X_smote[palabras_mas_usadas_smote1].reindex(X_smote.index)

# Ajuste del modelo
param_grid = {
    'n_estimators': [100],
    'max_depth': [5]
}
clasificador = GradientBoostingClassifier(random_state=0)
grid = GridSearchCV(clasificador, param_grid, cv=2)
grid.fit(X_smote, Y_smote)
clasificador_final = grid.best_estimator_

# Asignar el vectorizador y escalador al módulo de preprocesamiento
set_vectorizer(bv)
set_scaler(scaler)

# Crear el pipeline
pipeline = Pipeline([
    ('text_preprocessing', FunctionTransformer(text_preprocessing_function)),
    ('vectorization', FunctionTransformer(vectorization_function)),
    ('scaling', FunctionTransformer(scaling_function)),
    ('classification', clasificador_final)
])

# Guardar el pipeline entrenado
dump(pipeline, 'pipeline.joblib')

# (Opcional) Realizar una prueba con los datos de test
datatest["message"] = (datatest["Titulo"] + " " + datatest["Descripcion"]).astype(str)
datatest = datatest.drop(columns=["ID", "Titulo", "Descripcion", "Fecha"])
y_pred_test = pipeline.predict(datatest["message"])
print("Predicciones test:", y_pred_test)
datatest['Label'] = y_pred_test
datatest.to_csv('particion_prueba_estudiantes.csv', index=False)
