
# import libraries
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from numpy import random
random.seed(17)

def feature_importance(df, target):
    """plotea las variables que más influen en el modelo, de mayor a menor
        recoge un dataframe y un target en formato string
        entrena un RandomForest y plotea las variables"""

    # se definen los parametros para entrenar el modelo
    params = {'random_state': 42, 'n_jobs': 4, 'n_estimators': 100, 'max_depth': 4}
    # se entrena un RandomForest y se plotean las variables que tuvieron más importancia en el modelo
    y = df[target]
    x = df.drop(target, axis=1)
    # entrena un RandomForest Classifier
    clf = RandomForestClassifier(**params)
    clf = clf.fit(x, y)
    # Plotea las 15 mejores Features importances
    features = clf.feature_importances_[:10]
    columnas = x.columns[:10]
    imp = pd.Series(data=features, index=columnas).sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    plt.title("Feature importance")
    ax = sns.barplot(y=imp.index, x=imp.values, palette="Blues_d", orient='h')
    plt.show()

def grid_search(X_train, y_train):
    """encuentra los parametros optimos del modelo con un gridsearch
        recoge un set de entrenamiento con las features y un set con las etiquetas del target
        imprime los parametros"""

    parameters = {'criterion': ['gini', 'entropy'],
                  'n_estimators': [100, 300, 500, 1000],
                  'max_features': ['sqrt', 'log2', None],
                  'max_depth': [None, 2, 3],
                  'max_leaf_nodes': [None, 2, 3]}
    rf_hp = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf_hp,
                           parameters,
                           cv=2,
                           verbose=True)
    rf_grid.fit(X_train, y_train)
    print(rf_grid.best_estimator_)


if __name__ == '__main__':
    # import data
    df_train = pd.read_csv("datasets/train.csv", sep=';')
    df_test = pd.read_csv("datasets/test.csv", sep=';')

    # visualización general
    print(df_train.head())
    print(df_test.head())

    # información general
    print(df_train.info())
    print(df_test.info())

    # estadísticas básicas
    print(df_train.describe().T)
    print(df_test.describe().T)

    # definición de las features
    features = df_test.columns

    # visualización de la repartición de las features
    for feature in features:
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.kdeplot(x=feature, data=df_train)
        ax.set_title('kde ' + feature)
        plt.show()

    # visualización de la repartición del target
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.countplot(x='target', data=df_train)
    ax.set_title('countplot target')
    plt.show()

    # porcentaje de cada label del target
    reparticion_label = df_train['target'].value_counts(normalize=True).mul(100)
    print(reparticion_label)

    # la repartición es casí perfecta: 33% cada label. No se tendrá que compensar para modelizar

    # heatmap de correlaciones
    plt.rcParams['figure.figsize'] = 12, 8
    sns.heatmap(df_train.corr(), annot=True, cmap="YlGnBu")
    plt.show()

    # las features 7, 8 y 4 tienen muy poca correlación con el target (de menor a mayor). No influirán mucho en el modelo.

    # gráfico de la importancia de las variables para el modelo
    feature_importance(df_train, 'target')

    # se confirma que las features 7, 8 y 4 son las que menos importancia tiene a la hora de modelizar
    # df_train.drop(['feature4', 'feature8', 'feature7'], axis=1, inplace=True)
    # después de intentar eliminando estas features, se decide quedarlas porque empeora el f1_score 2 puntos porcentuales

    # split train/test del df_train
    # se guarda un 30% de datos para el test y un 70% para el train
    y = df_train['target']
    X = df_train.drop('target', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # las variables estan estandarizadas. Para ponerlas todas en la misma escala de -1 a 1 se van a noramlizar también
    scaler = Normalizer()
    X_train = scaler.fit_transform(X_train)  # noramliza y define la regla de normalización
    X_test = scaler.transform(X_test)        # normaliza según la regla definida con el X_train
    df_test = scaler.transform(df_test)      # normaliza el df_test también según la regla definida con el X_train

    # entreno del modelo
    rf_base = RandomForestClassifier(random_state=42)
    rf_base.fit(X_train, y_train)
    y_pred = rf_base.predict(X_test)
    f1_score = metrics.f1_score(y_test, y_pred, average='macro')
    print(f1_score)

    # sale un f1_score de 0.91204 con los parametros por defectos

    # gridsearch para hiperparametrizar el modelo
    # se comenta esta linea para ahorrar tiempo una vez se encontraron los parametros optimos
    # grid_search(X_train, y_train)

    # después del gridsearch, volvemos a entrenar el modelo con los nuevos parametros.
    rf_hp = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf_hp.fit(X_train, y_train)
    y_pred = rf_hp.predict(X_test)
    f1_score = metrics.f1_score(y_test, y_pred, average='macro')
    print(f1_score)

    # el f1_score subió a 0.91536. Se ganan 3 decimas porcentuales

    # predicción
    pred = rf_hp.predict(df_test)

    # se crea un dataframe con la predicción
    df_pred = pd.DataFrame(pred, columns=['prediction'])

    # se exporta a csv
    df_pred.to_csv('predictions.csv', index=False)

    # se exporta a json
    df_pred.to_json('predictions.json')
