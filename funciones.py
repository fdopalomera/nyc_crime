import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report



# Gráfico distribuciones de variables

"""
Grafica la distribución de las variables dentro de un dataframe, según su tipo de dato.

"""

def distribution_plots(df, columns=3):
    
    rows = np.ceil(df.shape[1] / columns)
    height = rows * 3.5
    fig = plt.figure(figsize=(12, height))
 
    for n, i in enumerate(df.columns):
        
        if df[i].dtype in ('object', 'int64') :
            fig.add_subplot(rows, columns, n+1)
            ax = sns.countplot(x=i, data=df)
            plt.title(i)
            plt.xlabel('')
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x()+p.get_width()/2., height + .5,
                    '{:1.2f}'.format(height/len(df[i])), ha="center")

            
        if df[i].dtype == 'float64':
            fig.add_subplot(rows, columns, n+1)
            ax = sns.distplot(df[i])
            plt.title(i)
            plt.xlabel('')
            
    plt.tight_layout()

    return

# Función de preprocesamiento de base

def pre_processing(df, num_cols, obj_cols, exclude, outliers_cols, target, 
                    drop_nan=False, remove_outliers=False, std_scaler=False, 
                    ord_encoder=True, custom_split=True):
    """
    Función para aplicar un preprocesamiento de datos sobre un dataframe

    """

    columns = num_cols+obj_cols+exclude
    tmp = df[columns]
    
    # Remoción de filas con datos perdidos
    if drop_nan:
        tmp = tmp.dropna()

   # Detección y eliminación de outliers
    if remove_outliers:
     
        Q1 = tmp[outliers_cols].quantile(0.25)
        Q3 = tmp[outliers_cols].quantile(0.75)
        IQR = Q3 - Q1
        tmp = tmp[~((tmp < (Q1 - 1.5 * IQR)) |(tmp > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    num_steps, obj_steps = None, None
    
    if std_scaler:
        num_steps = StandardScaler()
    if ord_encoder:
        obj_steps = OrdinalEncoder()

    # Creación de pipeline para preproceso
    num_pipe = make_pipeline(num_steps)
    obj_pipe = make_pipeline(obj_steps)

    column_transformer = make_column_transformer(
                            (num_pipe, num_cols),
                            (obj_pipe, obj_cols),
                            ('passthrough', exclude))

    preprocessed = column_transformer.fit_transform(tmp)

    df_pre = pd.DataFrame(data=preprocessed, columns=columns)


     # Train test split
    if custom_split:
   
        df_train = df_pre[df_pre['sample'] == 'train']
        df_test = df_pre[df_pre['sample'] == 'test']

        X_train = df_train.drop(columns=[target, 'sample'])
        y_train = df_train[target]

        X_test = df_test.drop(columns=[target, 'sample'])
        y_test = df_test[target]
    
    else:
        X = df_pre.drop(columns=[target, 'sample'])
        y = df.pre[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=123)

    return X_train, X_test, y_train, y_test


# Métricas de problema de clasificación

def clf_metrics(clf, X_train, y_train, X_test, y_test):
    """
    Imprime un reporte con las métricas de problemas de clasificación clásicas:

    """    
    tic = time()
    # Entrenar el modelo
    clf.fit(X_train, y_train)
    # Imprimir mejores parámetros
    print(clf.best_params_)
    # Predecir la muestra de validación
    y_hat = clf.predict(X_test)
    # Métricas
    metrics = {'ROC_Score': roc_auc_score(y_test, y_hat).round(3),
               'Confusion_Matrix': confusion_matrix(y_test, y_hat).round(3),
               'Classification_Report': classification_report(y_test, y_hat)}
    for key, value in metrics.items():
        print('{}:\n{}'.format(key, value))
    return print("Realizado en {:.3f}s".format(time() - tic))