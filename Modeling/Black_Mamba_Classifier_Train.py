#!/usr/bin/env python
# coding: utf-8

## Importamos las librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from keras import optimizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adagrad, SGD
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

## Indicamos el path
path = "C:/Users/mibra/Desktop/NBA/"

## Leemos los archivos
dat = pd.read_csv(path + "final_dat.csv", sep=";", header=0, encoding='latin-1')

## Imprimimos el tamaño del set de datos final y observamos una muestra
print("\nEl tamaño del data set es:", dat.shape)

## Observamos la distribución del target en train y test
pd.DataFrame(dat.groupby(["dataset", "w/l"])["w/l"].aggregate("count"))


## Contamos los posibles duplicados y los eliminamos
duplicated_rows = dat.duplicated(subset='match_up').sum()   
if (duplicated_rows > 0):
    dat = dat.drop_duplicates(subset='match_up', keep='last').reset_index(drop=True)
    print("Número de filas duplicadas eliminadas:", duplicated_rows)
else:
    print("No se han encontrado filas duplicadas")


## Fijamos las variables para recuperar la predicción
fixed = dat.iloc[:, 0:5]

## Obtenemos los set de train y test
X_train = dat.loc[dat["dataset"]=="train", :].drop(["dataset", "match_up", "date", "w/l", "target"], axis=1).values
y_train = dat.loc[dat["dataset"]=="train", "w/l"].values

X_val = dat.loc[dat["dataset"]=="val", :].drop(["dataset", "match_up", "date", "w/l", "target"], axis=1).values
y_val = dat.loc[dat["dataset"]=="val", "w/l"].values

X_test = dat.loc[dat["dataset"]=="test", :].drop(["dataset", "match_up", "date", "w/l", "target"], axis=1).values
y_test = dat.loc[dat["dataset"]=="test", "w/l"].values

## Imprimimos el tamaño de cada set de datos
print("El tamaño del set de train es:", X_train.shape)
print("El tamaño del target de train es:", y_train.shape)

print("\nEl tamaño del set de val es:", X_val.shape)
print("El tamaño del target de val es:", y_val.shape)

print("\nEl tamaño del set de test es:", X_test.shape)
print("El tamaño del target de test es:", y_test.shape)


# ## Machine Learning Models
# 
# [Documentación AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html?highlight=ada#sklearn.ensemble.AdaBoostClassifier)
# 
# [Documentación Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random%20forest#sklearn.ensemble.RandomForestClassifier)
# 
# [Documentación XGBoost](https://xgboost.readthedocs.io/en/latest/parameter.html)
# 
# [Documentación Parametros XGBoost](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)
# 
# [Documentación LGBM](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
# 
# [Documentación Parametros LGBM](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc)
# 
# [Documentación SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
# 
# [Documentación Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB)
# 
# [Documentación KNeighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)

## Cargamos los parametros del modelo optimizado
ab_params = pd.read_csv(path+"models/AdaBoost_Grid_Results.csv", sep=";", encoding='utf-8').iloc[0, 0:-5].to_dict()
rf_params = pd.read_csv(path+"models/Random_Forest_Grid_Results.csv", sep=";").iloc[0, 0:-5].to_dict()

xgb_params = pd.read_csv(path+"models/XGBoost_Grid_Results.csv", sep=";").iloc[0, 0:-5].to_dict()
xgb_params['n_estimators'] = int(xgb_params['n_estimators'])

lgbm_params = pd.read_csv(path+"models/LGBM_Grid_Results.csv", sep=";").iloc[0, 0:-5].to_dict()
lgbm_params['n_estimators'] = int(lgbm_params['n_estimators'])
lgbm_params['min_data_in_leaf'] = int(lgbm_params['min_data_in_leaf'])
lgbm_params['num_leaves'] = int(lgbm_params['num_leaves'])

svc_params = pd.read_csv(path+"models/SVC_Grid_Results.csv", sep=";").iloc[0, 0:-5].to_dict()
nb_params = pd.read_csv(path+"models/NB_Grid_Results.csv", sep=";").iloc[0, 0:-5].to_dict()
kn_params = pd.read_csv(path+"models/KNeighbors_Grid_Results.csv", sep=";").iloc[0, 0:-5].to_dict()

## Definimos los parametros de cada modelo
ab = AdaBoostClassifier(**ab_params).fit(X_train, y_train)
rf = RandomForestClassifier(**rf_params).fit(X_train, y_train)
xgb = XGBClassifier(**xgb_params).fit(X_train, y_train)
lgbm = LGBMClassifier(**lgbm_params).fit(X_train, y_train)
svc = SVC(**svc_params).fit(X_train, y_train)
nb = BernoulliNB(**nb_params).fit(X_train, y_train)
kn = KNeighborsClassifier(**kn_params).fit(X_train, y_train)

## Generamos un listado de los modelos y predecimos el set entero
model_list = [(ab, "AdaBoost"), (rf, "Random Forest"),
              (xgb, "XGBoost"), (lgbm, "LGBM"), (svc, "SVC"),
              (nb, "Naive Bayes"), (kn, "KNeighbors")]

predictions = pd.DataFrame()
for model, name in model_list:  
    print("El accuracy de {} es {:.2f}%\n".format(name, 100*model.score(X_val, y_val))) 
    pred = pd.DataFrame(model.predict(np.concatenate([X_train, X_val, X_test])), columns=["{}_Pred".format(name)])
    predictions = pd.concat([predictions, pred], axis=1)
predictions = pd.concat([fixed, predictions], axis=1)

# Guardamos las predicciones
predictions.to_csv(path+"/models/All_Pred_Train.csv", sep=";", header=True, index=False)


# ## Deep Stacking Model

## Generamos una función para típificar las predicciones del ensemble
def tipify(data):
    ## Fijamos las predicciones
    fixed = data.iloc[:, :5]
    data = data.iloc[:, 5:]
    
    ## Obtenemos los vectores de media y desviación típica
    m = data.mean(axis=0)
    s = data.std(axis=0)

    ## Tipificamos cada variable
    for i in range(0, data.shape[1]):
        data.iloc[:, i] = (data.iloc[:, i]-m[i])/s[i]
    data = pd.concat([fixed, data], axis=1)
    return (data, m, s)


## Tipificamos las predicciones
predictions, m, s = tipify(predictions)

## Imprimimos el tamaño de cada set de datos
print("\nEl tamaño del vector de medias es:", m.shape)
print("El tamaño del vector de desviaciones típicas es:", s.shape)

## Juntamos todas las predicciones en una tabla
dat = reduce(lambda x,y: pd.merge(x,y, on=['match_up','date','w/l','target','dataset']),[dat, predictions])

## Obtenemos los set de train y test
X_train = dat.loc[dat["dataset"]=="train", :].drop(["dataset", "match_up", "date", "w/l", "target"], axis=1).values
y_train = dat.loc[dat["dataset"]=="train", "w/l"].values

X_val = dat.loc[dat["dataset"]=="val", :].drop(["dataset", "match_up", "date", "w/l", "target"], axis=1).values
y_val = dat.loc[dat["dataset"]=="val", "w/l"].values

X_test = dat.loc[dat["dataset"]=="test", :].drop(["dataset", "match_up", "date", "w/l", "target"], axis=1).values
y_test = dat.loc[dat["dataset"]=="test", "w/l"].values

## Imprimimos el tamaño de cada set de datos
print("\nEl tamaño del set de train es:", X_train.shape)
print("El tamaño del target de train es:", y_train.shape)

print("\nEl tamaño del set de val es:", X_val.shape)
print("El tamaño del target de val es:", y_val.shape)

print("\nEl tamaño del set de test es:", X_test.shape)
print("El tamaño del target de test es:", y_test.shape)


## Cargamos los valores del grid de la red neuronal
nn_grid = pd.read_csv(path+'/models/NeuralNet_Grid_Results.csv', sep=";").iloc[0, 0:-5].to_dict()
rate = nn_grid['rate']
neurons = nn_grid['units']
optimizer = nn_grid['optimizer']
hidden_layers = nn_grid['hidden_layers']
batch_size = nn_grid['batch_size']

## Generamos un modelo secuencial
model = Sequential()
model.add(Dense(neurons, input_shape=[X_train.shape[1]], activation="tanh"))

for i in range(hidden_layers):
    model.add(Dropout(rate))
    model.add(Dense(neurons, activation="tanh"))

model.add(Dropout(rate))
model.add(Dense(1, activation="sigmoid"))

## Compilamos la función de coste, las métricas y el optimizador
if optimizer == "Adagrad":
    opt = Adagrad(lr=0.001)
elif optimizer == "SGD":
    opt = SGD(lr=0.001)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["acc"])

## Mostramos un resumen de la estructura de la red neuronal
model.summary()


# Fijamos un callback para generrar un early_stopping y guardar el mejor modelo
callbacks = [EarlyStopping(monitor='val_loss', patience=50),
             ModelCheckpoint(filepath=path + "best_classifier_nba.h5", monitor='val_loss', save_best_only=True)]

# Entrenar el modelo
history = model.fit(x=X_train,
                    y=y_train,
                    epochs=1000,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks)

## Agrupamos el entrenamiento en un dataframe
hist = pd.DataFrame(history.history)

## Añadimos cada epcoh al dataframe
hist['Epoch'] = history.epoch

## Transformamos el dataframe para poder plotear los resultados
df = hist.melt(id_vars='Epoch',
               var_name='Type',
               value_name='Error',
               value_vars=['loss','val_loss', 'acc', 'val_acc'])

## Visualizamos el entrenamiento del modelo
fig, ax = plt.subplots(figsize=(16, 6))
_ = sns.lineplot(x='Epoch', y='Error', hue='Type', data=df)
plt.title('Training Model History')

## Cargamos el modelo del checkpoint
model = load_model(path + "best_classifier_nba.h5")

## Obtener la precisión del modelo
train_results = model.evaluate(X_train, y_train, verbose=0)
val_results = model.evaluate(X_val, y_val, verbose=0)
test_results = model.evaluate(X_test, y_test, verbose=0)

## Agrupamos los resultados del training
results = pd.DataFrame({"Metrica":model.metrics_names, "Train": train_results, "Val": val_results, "Test": test_results})
results = results.round(4)

## Predecimos los datos de train y test
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

y_train_class_pred = model.predict_classes(X_train)
y_val_class_pred = model.predict_classes(X_val)
y_test_class_pred = model.predict_classes(X_test)

# ## Guardamos las métricas obtenidas
results.to_csv(path+"Black_Mamba_Classifier_Metrics.csv", sep=";", header=True, index=False, decimal=",")
print(results)

## Generamos una figura
plt.figure(figsize=(12, 8))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

## Representamos la diagonal
ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

# Obtenemos el Brier Score
clf_score = brier_score_loss(y_val, y_val_pred, pos_label=1)
    
# Obtenemos la curva de calibración
fraction_of_positives, mean_predicted_value = calibration_curve(y_val, y_val_pred, n_bins=10)
    
## Visualizamos la curva de calibración y su histograma
ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="{}: {:.2f}".format("Neural Net", clf_score))
ax2.hist(y_val_pred, range=(0, 1), bins=10, label="Neural Net", histtype="step", lw=2)

## Indicamos las labels y limites de la figura
ax1.set_ylabel("Fraction of positives")
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots (reliability curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper right")

plt.tight_layout()


## Unimos las predicciones de ambos set de datos junto con las variables fijadas
predictions = pd.DataFrame({"prediction": np.concatenate([y_train_class_pred[:,0], y_val_class_pred[:,0], y_test_class_pred[:,0]], axis=0),
                            "probability": np.concatenate([y_train_pred[:,0], y_val_pred[:,0], y_test_pred[:,0]], axis=0).round(2)})
results_dat = pd.concat([fixed, predictions], axis=1)
results_dat["w/l"] = results_dat["w/l"].astype("int64")

## Recuperamos los equipos de la key
results_dat["team1"] = results_dat['match_up'].str[15:22].str.split("@", expand=True)[0]
results_dat["team2"] = results_dat['match_up'].str[15:22].str.split("@", expand=True)[1]

## Guardamos las predicciones obtenidas
results_dat.to_csv(path+"Black_Mamba_Classifier_Predictions.csv", sep=";", header=True, index=False)


## Generamos una lista con las diferentes clases
labels = ["0", "1"]

## Evaluamos los modelos con diversos metodos (Precision, Recall, F1) 
report = classification_report(y_true=y_test, y_pred=y_test_class_pred, target_names=labels)

print("Evaluación del modelo:\n", report)


## Cargamos la función para visualizar la matriz de confusión
def plot_confusion_matrix(cm, title="Matriz de Confusión", cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, np.arange(cm.shape[0]))
    plt.yticks(tick_marks, np.arange(cm.shape[0]))
    plt.tight_layout()
    plt.ylabel("Clase Real")
    plt.xlabel("Clase Estimada")
    
## Calculamos la matriz de confusión
cm = confusion_matrix(y_test, y_test_class_pred)
print(cm, "\n")
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm.round(3))

## Visualizamos la matriz de confusión de cada modelo
fig, ax= plt.subplots(figsize=(5, 5))
plot_confusion_matrix(cm)
plt.show()

## Cargamos la función para visualizar la curva roc
def plot_roc_curve(fpr,tpr): 
    ## Obtenemos el AUC
    auc_metric = auc(fpr, tpr)
    print("AUC Value:", auc_metric.round(4))
    
    ## Generamos la visualización
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(fpr, tpr, color='blue') 
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0,1, 0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.title('ROC curve per class')
    plt.show()

## Calculamos los elementos de la curva roc
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)    

## Ploteamos la curva roc
plot_roc_curve(fpr, tpr) 


# ## Final Model Training

## Leemos los archivos
dat = pd.read_csv(path + "final_dat.csv", sep=";", header=0, encoding='latin-1')

## Obtenemos el set de train final 
X_train = dat.drop(["dataset", "match_up", "date", "w/l", "target"], axis=1).values
y_train = dat.loc[:, "w/l"].values

## Imprimimos el tamaño de cada set de datos
print("\nEl tamaño del set de train final es:", X_train.shape)
print("El tamaño del target de train final es:", y_train.shape)


## Cargamos los parametros del modelo optimizado
ab_params = pd.read_csv(path+"models/AdaBoost_Grid_Results.csv", sep=";", encoding='utf-8').iloc[0, 0:-5].to_dict()
rf_params = pd.read_csv(path+"models/Random_Forest_Grid_Results.csv", sep=";").iloc[0, 0:-5].to_dict()

xgb_params = pd.read_csv(path+"models/XGBoost_Grid_Results.csv", sep=";").iloc[0, 0:-5].to_dict()
xgb_params['n_estimators'] = int(xgb_params['n_estimators'])

lgbm_params = pd.read_csv(path+"models/LGBM_Grid_Results.csv", sep=";").iloc[0, 0:-5].to_dict()
lgbm_params['n_estimators'] = int(lgbm_params['n_estimators'])
lgbm_params['min_data_in_leaf'] = int(lgbm_params['min_data_in_leaf'])
lgbm_params['num_leaves'] = int(lgbm_params['num_leaves'])

svc_params = pd.read_csv(path+"models/SVC_Grid_Results.csv", sep=";").iloc[0, 0:-5].to_dict()
nb_params = pd.read_csv(path+"models/NB_Grid_Results.csv", sep=";").iloc[0, 0:-5].to_dict()
kn_params = pd.read_csv(path+"models/KNeighbors_Grid_Results.csv", sep=";").iloc[0, 0:-5].to_dict()

## Definimos los parametros de cada modelo
ab = AdaBoostClassifier(**ab_params).fit(X_train, y_train)
rf = RandomForestClassifier(**rf_params).fit(X_train, y_train)
xgb = XGBClassifier(**xgb_params).fit(X_train, y_train)
lgbm = LGBMClassifier(**lgbm_params).fit(X_train, y_train)
svc = SVC(**svc_params).fit(X_train, y_train)
nb = BernoulliNB(**nb_params).fit(X_train, y_train)
kn = KNeighborsClassifier(**kn_params).fit(X_train, y_train)

## Generamos un listado de los modelos y predecimos el set entero
model_list = [(ab, "AdaBoost"), (rf, "Random Forest"),
              (xgb, "XGBoost"), (lgbm, "LGBM"), (svc, "SVC"),
              (nb, "Naive Bayes"), (kn, "KNeighbors")]

predictions = pd.DataFrame()
for clf, name in model_list:
    pickle.dump(clf, open(path+'/models/{}_model_train.pkl'.format(name), 'wb'))
    pred = pd.DataFrame(clf.predict(X_train), columns=["{}_Pred".format(name)])
    predictions = pd.concat([predictions, pred], axis=1)
predictions = pd.concat([fixed, predictions], axis=1)


## Tipificamos las predicciones
predictions, m, s = tipify(predictions)

## Guardamos los vectores de media y desviación típica
m.to_csv(path+"/models/mean_vector_train.csv", sep=";", header=False, index=True)
s.to_csv(path+"/models/sd_vector_train.csv", sep=";", header=False, index=True)

## Imprimimos el tamaño de cada set de datos
print("\nEl tamaño del vector de medias es:", m.shape)
print("El tamaño del vector de desviaciones típicas es:", s.shape)

## Juntamos todas las predicciones en una tabla
dat = reduce(lambda x,y: pd.merge(x,y, on=['match_up','date','w/l','target','dataset']),[dat, predictions])

## Obtenemos los set de train y test
X_train = dat.drop(["dataset", "match_up", "date", "w/l", "target"], axis=1).values
y_train = dat.loc[:, "w/l"].values

## Imprimimos el tamaño de cada set de datos
print("\nEl tamaño del set de train es:", X_train.shape)
print("El tamaño del target de train es:", y_train.shape)


## Cargamos los valores del grid de la red neuronal
nn_grid = pd.read_csv(path+'/models/NeuralNet_Grid_Results.csv', sep=";").iloc[0, 0:-5].to_dict()
rate = nn_grid['rate']
neurons = nn_grid['units']
optimizer = nn_grid['optimizer']
hidden_layers = nn_grid['hidden_layers']
batch_size = nn_grid['batch_size']

## Generamos un modelo secuencial
model = Sequential()
model.add(Dense(neurons, input_shape=[X_train.shape[1]], activation="tanh"))

for i in range(hidden_layers):
    model.add(Dropout(rate))
    model.add(Dense(neurons, activation="tanh"))

model.add(Dropout(rate))
model.add(Dense(1, activation="sigmoid"))

## Compilamos la función de coste, las métricas y el optimizador
if optimizer == "Adagrad":
    opt = Adagrad(lr=0.001)
elif optimizer == "SGD":
    opt = SGD(lr=0.001)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["acc"])

## Mostramos un resumen de la estructura de la red neuronal
model.summary()


# Fijamos un callback para generrar un early_stopping y guardar el mejor modelo
callbacks = [EarlyStopping(monitor='val_loss', patience=50),
             ModelCheckpoint(filepath=path + "Final_Black_Mamba_Model.h5", monitor='val_loss', save_best_only=True)]

# Entrenar el modelo
model.fit(x=X_train,
          y=y_train,
          epochs=1000,
          batch_size=batch_size,
          validation_split=0.05,
          callbacks=callbacks)
