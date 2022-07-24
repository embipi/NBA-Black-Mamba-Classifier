#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Importamos las librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adagrad

from sklearn.model_selection import GridSearchCV
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


# In[2]:


## Indicamos el path
path = "C:/Users/mibra/Desktop/NBA/"

## Leemos los archivos
dat = pd.read_csv(path + "final_dat.csv", sep=";", header=0, encoding='latin-1')

## Imprimimos el tamaño del set de datos final y observamos una muestra
print("\nEl tamaño del data set es:", dat.shape)

## Observamos la distribución del target en train y test
pd.DataFrame(dat.groupby(["dataset", "w/l"])["w/l"].aggregate("count"))


# In[3]:


## Contamos los posibles duplicados y los eliminamos
duplicated_rows = dat.duplicated(subset='match_up').sum()   
if (duplicated_rows > 0):
    dat = dat.drop_duplicates(subset='match_up', keep='last').reset_index(drop=True)
    print("Número de filas duplicadas eliminadas:", duplicated_rows)
else:
    print("No se han encontrado filas duplicadas")


# In[4]:


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

# ### AdaBoost
# [Documentación AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html?highlight=ada#sklearn.ensemble.AdaBoostClassifier)

# In[5]:


## Execution time
t0 = time.time()

## Valores del grid
tuned_n_estimators = np.array([120, 180])
tuned_learning_rate = np.array([0.1, 0.15])
tuned_algorithm = np.array(["SAMME", "SAMME.R"])

## Agrupamos los valores
tuned_parameters = [{"n_estimators":tuned_n_estimators,
                     "learning_rate":tuned_learning_rate,
                     "algorithm": tuned_algorithm}]

## Modelo AdaBoost optimizado
ab_grid = GridSearchCV(AdaBoostClassifier(), tuned_parameters, cv=10, iid=True, return_train_score=True).fit(X_train, y_train)

## Guardamos el mejor modelo del grid
ab = ab_grid.best_estimator_
pickle.dump(ab, open(path+'/models/AdaBoost_model.pkl', 'wb'))

## Obtenemos el accuracy con el set de test
ab_grid_acc = pd.DataFrame({"Accuracy":ab.score(X_val, y_val)}, index=[0])
print("\nEl accuracy del test con el modelo AdaBoost optimizado es %2.2f%%\n" %(100*ab_grid_acc.Accuracy)) 

## Guardamos el accuracy
ab_grid_acc.to_csv(path+"/models/AdaBoost_Accuracy.csv", sep=";", header=True, index=False)

## Generamos un dataframe con los resultados del grid
grid_results = pd.concat([pd.DataFrame(ab_grid.cv_results_["params"]),
                          pd.DataFrame(ab_grid.cv_results_["mean_test_score"].round(4), columns=["Test_acc"]),
                          pd.DataFrame(ab_grid.cv_results_["std_test_score"].round(4), columns=["Test_std"]),
                          pd.DataFrame(ab_grid.cv_results_["mean_train_score"].round(4), columns=["Train_acc"]),
                          pd.DataFrame(ab_grid.cv_results_["std_train_score"].round(4), columns=["Train_std"]),
                          pd.DataFrame(ab_grid.cv_results_["rank_test_score"], columns=["Ranking"])], axis=1)

grid_results = grid_results.sort_values('Ranking', ascending=True).reset_index(drop=True)

## Guardamos los resultados del grid
grid_results.to_csv(path+"/models/AdaBoost_Grid_Results.csv", sep=";", header=True, index=False)

## Predecimos todos los registros con el modelo 
ab_predictions = pd.DataFrame(ab.predict(np.concatenate([X_train, X_val, X_test])), columns=["AdaBoost_Pred"])

ab_predictions = pd.concat([fixed, ab_predictions], axis=1)

## Guardamos las predicciones
ab_predictions.to_csv(path+"/models/AdaBoost_Predictions.csv", sep=";", header=True, index=False)

## Execution time
t = time.time()
print("Total time:", round((t - t0)/3600,2), "horas")


# ### Random Forest
# [Documentación Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random%20forest#sklearn.ensemble.RandomForestClassifier)

# In[6]:


## Execution time
t0 = time.time()

## Valores del grid
tuned_n_estimators = np.array([400, 500])
tuned_criterion = np.array(["gini", "entropy"])
tuned_max_depth = np.array([8, 10])
tuned_min_samples_split = np.array([7, 9])
tuned_min_samples_leaf = np.array([7, 9])
tuned_max_features = np.array(["log2", "auto"])

## Agrupamos los valores
tuned_parameters = [{"n_estimators":tuned_n_estimators,
                     "criterion":tuned_criterion,
                     "max_depth":tuned_max_depth,
                     "min_samples_split":tuned_min_samples_split,
                     "min_samples_leaf":tuned_min_samples_leaf,
                     "max_features":tuned_max_features}]

## Modelo Random Forest optimizado
rf_grid = GridSearchCV(RandomForestClassifier(n_jobs=-1, bootstrap=True),
                       tuned_parameters, cv=10, iid=True, return_train_score=True).fit(X_train, y_train)

## Guardamos el mejor modelo del grid
rf = rf_grid.best_estimator_
pickle.dump(rf, open(path+'/models/Random_Forest_model.pkl', 'wb'))

## Obtenemos el accuracy con el set de test
rf_grid_acc = pd.DataFrame({"Accuracy":rf.score(X_val, y_val)}, index=[0])
print("\nEl accuracy del test con el modelo Random Forest optimizado es %2.2f%%\n" %(100*rf_grid_acc.Accuracy)) 

## Guardamos el accuracy
rf_grid_acc.to_csv(path+"/models/Random_Forest_Accuracy.csv", sep=";", header=True, index=False)

## Generamos un dataframe con los resultados del grid
grid_results = pd.concat([pd.DataFrame(rf_grid.cv_results_["params"]),
                          pd.DataFrame(rf_grid.cv_results_["mean_test_score"].round(4), columns=["Test_acc"]),
                          pd.DataFrame(rf_grid.cv_results_["std_test_score"].round(4), columns=["Test_std"]),
                          pd.DataFrame(rf_grid.cv_results_["mean_train_score"].round(4), columns=["Train_acc"]),
                          pd.DataFrame(rf_grid.cv_results_["std_train_score"].round(4), columns=["Train_std"]),
                          pd.DataFrame(rf_grid.cv_results_["rank_test_score"], columns=["Ranking"])], axis=1)

grid_results = grid_results.sort_values('Ranking', ascending=True).reset_index(drop=True)

## Guardamos los resultados del grid
grid_results.to_csv(path+"/models/Random_Forest_Grid_Results.csv", sep=";", header=True, index=False)

## Predecimos todos los registros con el modelo 
rf_predictions = pd.DataFrame(rf.predict(np.concatenate([X_train, X_val, X_test])), columns=["RandomForest_Pred"])

rf_predictions = pd.concat([fixed, rf_predictions], axis=1)

## Guardamos las predicciones
rf_predictions.to_csv(path+"/models/Random_Forest_Predictions.csv", sep=";", header=True, index=False)

## Execution time
t = time.time()
print("Total time:", round((t - t0)/3600, 2), "horas")


# ### XGBoost
# [Documentación XGBoost](https://xgboost.readthedocs.io/en/latest/parameter.html)
# 
# [Documentación Parametros](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)

# In[7]:


## Execution time
t0 = time.time()

## Valores del grid
tuned_n_estimators = np.array([100, 50])
tuned_eta = np.array([0.1, 0.15])
tuned_min_child_weight = np.array([0.6, 0.5])
tuned_gamma = np.array([0.1, 0.5])
tuned_colsample_bytree = np.array([0.45, 0.4])
tuned_lambda = np.array([0.3, 0.35])
tuned_alpha = np.array([0.3, 0.35])

## Agrupamos los valores
tuned_parameters = [{"n_estimators":tuned_n_estimators,
                     "eta":tuned_eta,
                     "min_child_weight":tuned_min_child_weight,
                     "gamma":tuned_gamma,
                     "colsample_bytree":tuned_colsample_bytree,
                     "lambda":tuned_lambda,
                     "alpha":tuned_alpha}]

## Modelo XGBoost optimizado
xgb_grid = GridSearchCV(XGBClassifier(objective='binary:logistic', eval_metric='error', n_jobs=-1, nthread=-1, max_depth=2,
                        subsample=0.8),tuned_parameters, cv=10, iid=True, return_train_score=True).fit(X_train, y_train)

## Guardamos el mejor modelo del grid
xgb = xgb_grid.best_estimator_
pickle.dump(xgb, open(path+'/models/XGBoost_model.pkl', 'wb'))

## Obtenemos el accuracy con el set de test
xgb_grid_acc = pd.DataFrame({"Accuracy":xgb.score(X_val, y_val)}, index=[0])
print("\nEl accuracy del test con el modelo XGBoost optimizado es %2.2f%%\n" %(100*xgb_grid_acc.Accuracy)) 

## Guardamos el accuracy
xgb_grid_acc.to_csv(path+"/models/XGBoost_Accuracy.csv", sep=";", header=True, index=False)

## Generamos un dataframe con los resultados del grid
grid_results = pd.concat([pd.DataFrame(xgb_grid.cv_results_["params"]),
                          pd.DataFrame(xgb_grid.cv_results_["mean_test_score"].round(4), columns=["Test_acc"]),
                          pd.DataFrame(xgb_grid.cv_results_["std_test_score"].round(4), columns=["Test_std"]),
                          pd.DataFrame(xgb_grid.cv_results_["mean_train_score"].round(4), columns=["Train_acc"]),
                          pd.DataFrame(xgb_grid.cv_results_["std_train_score"].round(4), columns=["Train_std"]),
                          pd.DataFrame(xgb_grid.cv_results_["rank_test_score"], columns=["Ranking"])], axis=1)

grid_results = grid_results.sort_values('Ranking', ascending=True).reset_index(drop=True)

## Guardamos los resultados del grid
grid_results.to_csv(path+"/models/XGBoost_Grid_Results.csv", sep=";", header=True, index=False)

## Predecimos todos los registros con el modelo 
xgb_predictions = pd.DataFrame(xgb.predict(np.concatenate([X_train, X_val, X_test])), columns=["XGBoost_Pred"])

xgb_predictions = pd.concat([fixed, xgb_predictions], axis=1)

## Guardamos las predicciones
xgb_predictions.to_csv(path+"/models/XGBoost_Predictions.csv", sep=";", header=True, index=False)

## Execution time
t = time.time()
print("Total time:", round((t - t0)/3600, 2), "horas")


# ### LGBM
# [Documentación LGBM](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
# 
# [Documentación Parametros](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc)

# In[8]:


## Execution time
t0 = time.time()

## Valores del grid
tuned_n_estimators  = np.array([200, 150])
tuned_learning_rate = np.array([0.05, 0.1])
tuned_num_leaves = np.array([28, 32])
tuned_min_data_in_leaf = np.array([18, 22])
tuned_lambda_l1 = np.array([0.1, 0.15])
tuned_lambda_l2 = np.array([0.3, 0.4])

## Agrupamos los valores
tuned_parameters = [{"n_estimators":tuned_n_estimators,
                     "learning_rate":tuned_learning_rate,
                     "num_leaves":tuned_num_leaves,
                     "min_data_in_leaf":tuned_min_data_in_leaf,
                     "lambda_l1":tuned_lambda_l1,
                     "lambda_l2":tuned_lambda_l2}]

## Modelo LGBM optimizado
lgbm_grid = GridSearchCV(LGBMClassifier(objective='binary', metric='binary_logloss', boosting='gbdt', max_depth=2),
                         tuned_parameters, cv=10, iid=True, return_train_score=True).fit(X_train, y_train)

## Guardamos el mejor modelo del grid
lgbm = lgbm_grid.best_estimator_
pickle.dump(lgbm, open(path+'/models/LGBM_model.pkl', 'wb'))

## Obtenemos el accuracy con el set de test
lgbm_grid_acc = pd.DataFrame({"Accuracy":lgbm.score(X_val, y_val)}, index=[0])
print("\nEl accuracy del test con el modelo LGBM optimizado es %2.2f%%\n" %(100*lgbm_grid_acc.Accuracy)) 

## Guardamos el accuracy
lgbm_grid_acc.to_csv(path+"/models/LGBM_Accuracy.csv", sep=";", header=True, index=False)

## Generamos un dataframe con los resultados del grid
grid_results = pd.concat([pd.DataFrame(lgbm_grid.cv_results_["params"]),
                          pd.DataFrame(lgbm_grid.cv_results_["mean_test_score"].round(4), columns=["Test_acc"]),
                          pd.DataFrame(lgbm_grid.cv_results_["std_test_score"].round(4), columns=["Test_std"]),
                          pd.DataFrame(lgbm_grid.cv_results_["mean_train_score"].round(4), columns=["Train_acc"]),
                          pd.DataFrame(lgbm_grid.cv_results_["std_train_score"].round(4), columns=["Train_std"]),
                          pd.DataFrame(lgbm_grid.cv_results_["rank_test_score"], columns=["Ranking"])], axis=1)

grid_results = grid_results.sort_values('Ranking', ascending=True).reset_index(drop=True)

## Guardamos los resultados del grid
grid_results.to_csv(path+"/models/LGBM_Grid_Results.csv", sep=";", header=True, index=False)

# Predecimos todos los registros con el modelo 
lgbm_predictions = pd.DataFrame(lgbm.predict(np.concatenate([X_train, X_val, X_test])), columns=["LGBM_Pred"])

lgbm_predictions = pd.concat([fixed, lgbm_predictions], axis=1)

## Guardamos las predicciones
lgbm_predictions.to_csv(path+"/models/LGBM_Predictions.csv", sep=";", header=True, index=False)

## Execution time
t = time.time()
print("Total time:", round((t - t0)/3600,2), "horas")


# ### SVC
# 
# [Documentación SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

# In[9]:


## Execution time
t0 = time.time()

## Valores del grid
tuned_C = np.array([0.3, 0.5])
tuned_gamma = np.array([0.001, 0.0005])

## Agrupamos los valores
tuned_parameters = [{"C":tuned_C,
                     "gamma":tuned_gamma}]

## Modelo svc optimizado
svc_grid = GridSearchCV(SVC(max_iter=-1, tol=0.001, kernel="rbf"),
                        tuned_parameters, cv=5, iid=True, return_train_score=True).fit(X_train, y_train)

## Guardamos el mejor modelo del grid
svc = svc_grid.best_estimator_
pickle.dump(svc, open(path+'/models/SVC_model.pkl', 'wb'))

## Obtenemos el accuracy con el set de test
svc_grid_acc = pd.DataFrame({"Accuracy":svc.score(X_val, y_val)}, index=[0])
print("\nEl accuracy del test con el modelo SVC optimizado es %2.2f%%\n" %(100*svc_grid_acc.Accuracy)) 

## Guardamos el accuracy
svc_grid_acc.to_csv(path+"/models/SVC_Accuracy.csv", sep=";", header=True, index=False)

## Generamos un dataframe con los resultados del grid
grid_results = pd.concat([pd.DataFrame(svc_grid.cv_results_["params"]),
                          pd.DataFrame(svc_grid.cv_results_["mean_test_score"].round(4), columns=["Test_acc"]),
                          pd.DataFrame(svc_grid.cv_results_["std_test_score"].round(4), columns=["Test_std"]),
                          pd.DataFrame(svc_grid.cv_results_["mean_train_score"].round(4), columns=["Train_acc"]),
                          pd.DataFrame(svc_grid.cv_results_["std_train_score"].round(4), columns=["Train_std"]),
                          pd.DataFrame(svc_grid.cv_results_["rank_test_score"], columns=["Ranking"])], axis=1)

grid_results = grid_results.sort_values('Ranking', ascending=True).reset_index(drop=True)

## Guardamos los resultados del grid
grid_results.to_csv(path+"/models/SVC_Grid_Results.csv", sep=";", header=True, index=False)

# Predecimos todos los registros con el modelo 
svc_predictions = pd.DataFrame(svc.predict(np.concatenate([X_train, X_val, X_test])), columns=["SVC_Pred"])

svc_predictions = pd.concat([fixed, svc_predictions], axis=1)

## Guardamos las predicciones
svc_predictions.to_csv(path+"/models/SVC_Predictions.csv", sep=";", header=True, index=False)

## Execution time
t = time.time()
print("Total time:", round((t - t0)/3600,2), "horas")


# ### Naive Bayes
# 
# [Documentación Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB)

# In[10]:


## Execution time
t0 = time.time()

## Valores del grid
tuned_alpha = np.array([1, 10, 100])
tuned_binarize = np.array([0, 0.001, 0.01])
tuned_fit_prior = np.array([True, False])

## Agrupamos los valores
tuned_parameters = [{"alpha":tuned_alpha,
                     "binarize":tuned_binarize,
                     "fit_prior":tuned_fit_prior}]

## Modelo svc optimizado
nb_grid = GridSearchCV(BernoulliNB(), tuned_parameters, cv=10, iid=True, return_train_score=True).fit(X_train, y_train)

## Guardamos el mejor modelo del grid
nb = nb_grid.best_estimator_
pickle.dump(nb, open(path+'/models/NB_model.pkl', 'wb'))

## Obtenemos el accuracy con el set de val
nb_acc = pd.DataFrame({"Accuracy":nb.score(X_val, y_val)}, index=[0])
print("\nEl accuracy del test con el modelo NB es %2.2f%%\n" %(100*nb_acc.Accuracy)) 

## Guardamos el accuracy
nb_acc.to_csv(path+"/models/NB_Accuracy.csv", sep=";", header=True, index=False)

## Generamos un dataframe con los resultados del grid
grid_results = pd.concat([pd.DataFrame(nb_grid.cv_results_["params"]),
                          pd.DataFrame(nb_grid.cv_results_["mean_test_score"].round(4), columns=["Test_acc"]),
                          pd.DataFrame(nb_grid.cv_results_["std_test_score"].round(4), columns=["Test_std"]),
                          pd.DataFrame(nb_grid.cv_results_["mean_train_score"].round(4), columns=["Train_acc"]),
                          pd.DataFrame(nb_grid.cv_results_["std_train_score"].round(4), columns=["Train_std"]),
                          pd.DataFrame(nb_grid.cv_results_["rank_test_score"], columns=["Ranking"])], axis=1)

grid_results = grid_results.sort_values('Ranking', ascending=True).reset_index(drop=True)

## Guardamos los resultados del grid
grid_results.to_csv(path+"/models/NB_Grid_Results.csv", sep=";", header=True, index=False)

# Predecimos todos los registros con el modelo 
nb_predictions = pd.DataFrame(nb.predict(np.concatenate([X_train, X_val, X_test])), columns=["NB_Pred"])

nb_predictions = pd.concat([fixed, nb_predictions], axis=1)

## Guardamos las predicciones
nb_predictions.to_csv(path+"/models/NB_Predictions.csv", sep=";", header=True, index=False)

## Execution time
t = time.time()
print("Total time:", round((t - t0)/3600,2), "horas")


# ### KNeighbors
# 
# [Documentación KNeighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)

# In[11]:


## Execution time
t0 = time.time()

## Valores del grid
tuned_n_neighbors = np.array([100, 200])
tuned_algorithm = np.array(["ball_tree", "kd_tree"])
tuned_leaf_size = np.array([26, 34])

## Agrupamos los valores
tuned_parameters = [{"n_neighbors":tuned_n_neighbors,
                     "algorithm":tuned_algorithm,
                     "leaf_size":tuned_leaf_size}]

## Modelo svc optimizado
kn_grid = GridSearchCV(KNeighborsClassifier(n_jobs=-1, weights="uniform", p=1),
                       tuned_parameters, cv=10, iid=True, return_train_score=True).fit(X_train, y_train)

## Guardamos el mejor modelo del grid
kn = kn_grid.best_estimator_
pickle.dump(kn, open(path+'/models/KNeighbors_model.pkl', 'wb'))

## Obtenemos el accuracy con el set de test
kn_grid_acc = pd.DataFrame({"Accuracy":kn.score(X_val, y_val)}, index=[0])
print("\nEl accuracy del test con el modelo KNeighbors optimizado es %2.2f%%\n" %(100*kn_grid_acc.Accuracy)) 

## Guardamos el accuracy
kn_grid_acc.to_csv(path+"/models/KNeighbors_Accuracy.csv", sep=";", header=True, index=False)

## Generamos un dataframe con los resultados del grid
grid_results = pd.concat([pd.DataFrame(kn_grid.cv_results_["params"]),
                          pd.DataFrame(kn_grid.cv_results_["mean_test_score"].round(4), columns=["Test_acc"]),
                          pd.DataFrame(kn_grid.cv_results_["std_test_score"].round(4), columns=["Test_std"]),
                          pd.DataFrame(kn_grid.cv_results_["mean_train_score"].round(4), columns=["Train_acc"]),
                          pd.DataFrame(kn_grid.cv_results_["std_train_score"].round(4), columns=["Train_std"]),
                          pd.DataFrame(kn_grid.cv_results_["rank_test_score"], columns=["Ranking"])], axis=1)

grid_results = grid_results.sort_values('Ranking', ascending=True).reset_index(drop=True)

## Guardamos los resultados del grid
grid_results.to_csv(path+"/models/KNeighbors_Grid_Results.csv", sep=";", header=True, index=False)

# Predecimos todos los registros con el modelo 
kn_predictions = pd.DataFrame(kn.predict(np.concatenate([X_train, X_val, X_test])), columns=["KNeighbors_Pred"])

kn_predictions = pd.concat([fixed, kn_predictions], axis=1)

## Guardamos las predicciones
kn_predictions.to_csv(path+"/models/KNeighbors_Predictions.csv", sep=";", header=True, index=False)

## Execution time
t = time.time()
print("Total time:", round((t - t0)/3600,2), "horas")


# ## Deep Stacking Model

# In[12]:


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


# In[13]:


## Cargamos todas las predicciones
ab_pred = pd.read_csv(path+'models/AdaBoost_Predictions.csv', sep=";", header=0)
rf_pred = pd.read_csv(path+'models/Random_Forest_Predictions.csv', sep=";", header=0)
xgb_pred = pd.read_csv(path+'models/XGBoost_Predictions.csv', sep=";", header=0)
lgbm_pred = pd.read_csv(path+'models/LGBM_Predictions.csv', sep=";", header=0)
svc_pred = pd.read_csv(path+'models/SVC_Predictions.csv', sep=";", header=0)
nb_pred = pd.read_csv(path+'models/NB_Predictions.csv', sep=";", header=0)
kn_pred = pd.read_csv(path+'models/KNeighbors_Predictions.csv', sep=";", header=0)

## Concatenamos todas las predicciones
predictions = reduce(lambda x,y: pd.merge(x,y, on=['match_up','date','w/l','target','dataset']),
                                         [ab_pred, rf_pred, xgb_pred, lgbm_pred, svc_pred, nb_pred, kn_pred])

print("Tamaño de las predicciones:", predictions.shape)

## Tipificamos las predicciones
predictions, m, s = tipify(predictions)

## Guardamos los vectores de media y desviación típica
m.to_csv(path+"/models/mean_vector_grid.csv", sep=";", header=False, index=True)
s.to_csv(path+"/models/sd_vector_grid.csv", sep=";", header=False, index=True)

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
print("El tamaño del set de val es:", X_val.shape)
print("El tamaño del set de test es:", X_test.shape)


# In[14]:


## Execution time
t0 = time.time()

## Generamos una función para el Wrapper de Scikit-Learn
def create_model(rate=0.65, units=150, optimizer='Adagrad', hidden_layers=1):
    ## Generamos un modelo secuencial
    model = Sequential()
    model.add(Dense(units, input_shape=[X_train.shape[1]], activation="tanh"))
    
    for i in range(hidden_layers):
        model.add(Dropout(rate))
        model.add(Dense(units, activation="tanh"))
    
    model.add(Dropout(rate))
    model.add(Dense(1, activation="sigmoid"))  
    
    ## Compilamos la función de coste, las métricas y el optimizador
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["acc"])
    return model

## Creamos el modelo
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=32, verbose=0)

## Valores del grid
tuned_units = np.array([150, 200])
tuned_rate = np.array([0.65, 0.55])
tuned_optimizer = np.array(["Adagrad", "SGD"])
tuned_hidden_layers = np.array([1, 2])
tuned_batch_size = np.array([32, 64])

## Agrupamos los valores
tuned_parameters = [{"units":tuned_units,
                     "rate":tuned_rate,
                     "optimizer": tuned_optimizer,
                     "batch_size":tuned_batch_size,
                     "hidden_layers":tuned_hidden_layers}]

## Generamos el grid
nn_grid = GridSearchCV(estimator=model, param_grid=tuned_parameters,
                       cv=5, iid=True, return_train_score=True).fit(X_train, y_train)

## Guardamos el mejor modelo del grid
nn = nn_grid.best_estimator_
# pickle.dump(nn, open(path+'/models/NeuralNet_model.pkl', 'wb'))

## Obtenemos el accuracy con el set de test
nn_grid_acc = pd.DataFrame({"Accuracy":nn.score(X_val, y_val)}, index=[0])
print("\nEl accuracy del test con el modelo Neural Net optimizado es %2.2f%%\n" %(100*nn_grid_acc.Accuracy)) 

## Guardamos el accuracy
nn_grid_acc.to_csv(path+"/models/NeuralNet_Accuracy.csv", sep=";", header=True, index=False)

## Generamos un dataframe con los resultados del grid
grid_results = pd.concat([pd.DataFrame(nn_grid.cv_results_["params"]),
                          pd.DataFrame(nn_grid.cv_results_["mean_test_score"].round(4), columns=["Test_acc"]),
                          pd.DataFrame(nn_grid.cv_results_["std_test_score"].round(4), columns=["Test_std"]),
                          pd.DataFrame(nn_grid.cv_results_["mean_train_score"].round(4), columns=["Train_acc"]),
                          pd.DataFrame(nn_grid.cv_results_["std_train_score"].round(4), columns=["Train_std"]),
                          pd.DataFrame(nn_grid.cv_results_["rank_test_score"], columns=["Ranking"])], axis=1)

grid_results = grid_results.sort_values('Ranking', ascending=True).reset_index(drop=True)

## Guardamos los resultados del grid
grid_results.to_csv(path+"/models/NeuralNet_Grid_Results.csv", sep=";", header=True, index=False)

## Execution time
t = time.time()
print("Total time:", round((t - t0)/3600,2), "horas")


# In[ ]:




