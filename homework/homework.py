# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#

import pandas as pd 
from sklearn.model_selection import GridSearchCV 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix, make_scorer
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import pickle
import numpy as np
import os
import json
import gzip
import time
import mlflow
import mlflow.sklearn

# Ignore warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#

def clean_data(data_df):
    df=data_df.copy()
    # Renombrar la columna "default payment next month" a "default"
    df=df.rename(columns={'default payment next month': 'default'})
    # Remover la columna "ID"
    df=df.drop(columns='ID')
    # Recodificar la variable EDUCATION: 0 es "NaN"
    df['EDUCATION'] = df['EDUCATION'].replace(0, np.nan)
    # Recodificar la variable MARRIAGE: 0 es "NaN"
    df['MARRIAGE'] = df['MARRIAGE'].replace(0, np.nan)
    # Eliminar los registros con informacion no disponible (es decir, con al menos una columna con valor nulo)
    df=df.dropna()
    # Agrupar los valores de EDUCATION > 4 en la categoria "others"
    df.loc[df['EDUCATION'] > 4, 'EDUCATION'] = 4
    return df


#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
def get_features_target(data, target_column):
    x = data.drop(columns=target_column)
    y = data[target_column]
    return x, y
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#

def create_pipeline(df):
    # Crear el pipeline
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    numerical_features = [col for col in df.columns if col not in categorical_features]

    # Definir los transformadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features),
            # ('num', StandardScaler(), numerical_features)
        ],
        remainder=StandardScaler()
    )

    # Definir el pipeline
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('select_k_best', SelectKBest(f_classif)),
            ('pca', PCA()),
            ('model', MLPClassifier(max_iter=15000))
        ]
    )

    return pipeline
#


# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#

def custom_scorer(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    return (precision + balanced_acc) / 2

custom_scorer = make_scorer(custom_scorer, greater_is_better=True)

def optimize_hyperparameters(pipeline, x_train, y_train):
    # param_grid = {
    #     'pca__n_components': [20],
    #     'select_k_best__k': [20],
    #     'model__hidden_layer_sizes': [(300,300,300,300,300)],
    #     'model__activation': ['relu'],
    #     'model__solver': ['adam'],
    #     'model__alpha': [0.667],
    #     # 'model__learning_rate': ['constant', 'adaptive'],
    #     'model__learning_rate_init': [0.0005],

    #     # 'model__class_weight': ['balanced', None]
    # }
    param_grid = {
        'pca__n_components': [20],
        'select_k_best__k': [20],
        'model__hidden_layer_sizes': [(35, 35, 30, 30, 30,30,30,30)],
        'model__activation': ['relu'],
        'model__solver': ['adam'],
        'model__alpha': [0.353],

        'model__learning_rate_init': [0.0005],

        # 'model__class_weight': ['balanced', None]
    }
    grid_search=GridSearchCV(pipeline, param_grid, cv=10, scoring='balanced_accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(x_train, y_train)
    
    with mlflow.start_run():

        grid_search.fit(x_train, y_train)
        
        # Log parameters
        best_params = grid_search.best_params_
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        
        # Log metrics
        y_pred_train = grid_search.predict(x_train)
        precision = precision_score(y_train, y_pred_train)
        balanced_acc = balanced_accuracy_score(y_train, y_pred_train)
        recall = recall_score(y_train, y_pred_train)
        f1 = f1_score(y_train, y_pred_train)
        
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("balanced_accuracy", balanced_acc)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log the model
        mlflow.sklearn.log_model(grid_search.best_estimator_, "model")
    
    return grid_search

# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#

def save_model(model):
    # If the models directory does not exist, create it
    if not os.path.exists('files/models'):
        os.makedirs('files/models')
    # Save the model using gzip
    with gzip.open('files/models/model.pkl.gz', 'wb') as f:
        pickle.dump(model, f)

#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#

def calculate_metrics(model, x_train, y_train, x_test, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    metrics_train = {
        'type': 'metrics',
        'dataset': 'train',
        'precision': float(precision_score(y_train, y_train_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_train, y_train_pred)),
        'recall': float(recall_score(y_train, y_train_pred)),
        'f1_score': float(f1_score(y_train, y_train_pred))
    }

    metrics_test = {
        'type': 'metrics',
        'dataset': 'test',
        'precision': float(precision_score(y_test, y_test_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, y_test_pred)),
        'recall': float(recall_score(y_test, y_test_pred)),
        'f1_score': float(f1_score(y_test, y_test_pred))
    }

    print(metrics_train)
    print(metrics_test)

    return metrics_train, metrics_test

#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

def calculate_confusion_matrix(model, x_train, y_train, x_test, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    cm_matrix_train = {
        'type': 'cm_matrix',
        'dataset': 'train',
        'true_0': {"predicted_0": int(cm_train[0, 0]), "predicted_1": int(cm_train[0, 1])},
        'true_1': {"predicted_0": int(cm_train[1, 0]), "predicted_1": int(cm_train[1, 1])}
    }

    cm_matrix_test = {
        'type': 'cm_matrix',
        'dataset': 'test',
        'true_0': {"predicted_0": int(cm_test[0, 0]), "predicted_1": int(cm_test[0, 1])},
        'true_1': {"predicted_0": int(cm_test[1, 0]), "predicted_1": int(cm_test[1, 1])}
    }

    return cm_matrix_train, cm_matrix_test

if __name__ == '__main__':
    
    # Carga de datos
    train_data_zip = 'files/input/train_data.csv.zip'
    test_data_zip = 'files/input/test_data.csv.zip'

    # Extraccion de los datos de los archivos zip
    train_data=pd.read_csv(
        train_data_zip,
        index_col=False,
        compression='zip')

    test_data=pd.read_csv(
        test_data_zip,
        index_col=False,
        compression='zip')
    
    # Limpieza de los datos
    train_data=clean_data(train_data)
    test_data=clean_data(test_data)

    # Dividir los datos en x_train, y_train, x_test, y_test
    x_train, y_train = get_features_target(train_data, 'default')
    x_test, y_test = get_features_target(test_data, 'default')

    # print(y_train.value_counts())

    # Crear el pipeline
    pipeline = create_pipeline(x_train)

    # Optimizar los hiperparametros
    start = time.time()
    model = optimize_hyperparameters(pipeline, x_train, y_train)
    end = time.time()
    print(f'Time to optimize hyperparameters: {end - start:.2f} seconds')

    print(model.best_params_)

    # Guardar el modelo
    save_model(model)

    # Calcular las metricas
    metrics_train, metrics_test = calculate_metrics(model, x_train, y_train, x_test, y_test)

    # Calcular las matrices de confusion
    cm_matrix_train, cm_matrix_test = calculate_confusion_matrix(model, x_train, y_train, x_test, y_test)

    print(cm_matrix_train)

    # Guardar las metricas

    # Crear la carpeta de output si no existe
    if not os.path.exists('files/output'):
        os.makedirs('files/output')

    # Guardar las metricas
    metrics = [metrics_train, metrics_test, cm_matrix_train, cm_matrix_test]
    pd.DataFrame(metrics).to_json('files/output/metrics.json', orient='records', lines=True)