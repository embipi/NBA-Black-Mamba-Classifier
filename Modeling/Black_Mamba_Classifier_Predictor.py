#!/usr/bin/env python
# coding: utf-8

## Importamos las librerias
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from selenium import webdriver
from fake_useragent import UserAgent 
from selenium.webdriver.chrome.options import Options
import time
from time import sleep
from functools import reduce
from tensorflow.keras.models import load_model
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

## Indicamos el path
path = "C:/Users/mibra/Desktop/NBA/"

## Leemos los archivos
dat = pd.read_csv(path + "final_future_games.csv", sep=";", header=0, encoding='latin-1')

## Fijamos las variables para recuperar la predicción
fixed = dat.iloc[:, 0:2]
data = dat.drop(["match_up", "date"], axis=1).values

## Imprimimos el tamaño del set de datos final y observamos una muestra
print("\nEl tamaño del data set es:", data.shape)


# ## Deep Stacking Model

## Cargamos los modelos 
ab = pickle.load(open(path+'/models/AdaBoost_model_train.pkl', 'rb'))
rf = pickle.load(open(path+'/models/Random Forest_model_train.pkl', 'rb'))
xgb = pickle.load(open(path+'/models/XGBoost_model_train.pkl', 'rb'))
lgbm = pickle.load(open(path+'/models/LGBM_model_train.pkl', 'rb'))
svc = pickle.load(open(path+'/models/SVC_model_train.pkl', 'rb'))
nb = pickle.load(open(path+'/models/Naive Bayes_model_train.pkl', 'rb'))
kn = pickle.load(open(path+'/models/KNeighbors_model_train.pkl', 'rb'))

## Generamos un listado de los modelos y predecimos el set entero
model_list = [(ab, "AdaBoost"), (rf, "Random Forest"),
              (xgb, "XGBoost"), (lgbm, "LGBM"), (svc, "SVC"),
              (nb, "Naive Bayes"), (kn, "KNeighbors")]

predictions = pd.DataFrame()
for model, name in model_list: 
    pred = pd.DataFrame(model.predict(data), columns=["{}_Pred".format(name)])
    predictions = pd.concat([predictions, pred], axis=1)
predictions = pd.concat([fixed, predictions], axis=1)

## Guardamos las predicciones
predictions.to_csv(path+"/models/All_Pred_Future_games.csv", sep=";", header=True, index=False)
predictions

## Generamos una función para típificar las predicciones del ensemble
def tipify_pred(data):
    ## Fijamos las predicciones
    fixed = data.iloc[:, :2]
    data = data.iloc[:, 2:]
    
    ## Obtenemos los vectores de media y desviación típica
    m = pd.read_csv(path+'models/mean_vector_train.csv', index_col=0, sep=";", header=None)
    s = pd.read_csv(path+'models/sd_vector_train.csv', index_col=0, sep=";", header=None)

    ## Tipificamos cada variable
    for i in range(0, data.shape[1]):
        data.iloc[:, i] = (data.iloc[:, i]-m.iloc[i, 0])/s.iloc[i, 0]
    data = pd.concat([fixed, data], axis=1)
    return data

## Cargamos todas las predicciones
pred = pd.read_csv(path+'models/All_Pred_Future_games.csv', sep=";", header=0)

print("Tamaño de las predicciones:", pred.shape)

## Tipificamos las predicciones
predictions = tipify_pred(pred)

## Juntamos todas las predicciones en una tabla
data = reduce(lambda x,y: pd.merge(x,y, on=['match_up','date']),[dat, predictions])

## Obtenemos el set de datos final
data = data.drop(["match_up", "date"], axis=1).values

## Imprimimos el tamaño de cada set de datos
print("\nEl tamaño del set es:", data.shape)


## Cargamos el modelo del checkpoint
model = load_model(path + "Final_Black_Mamba_Model.h5")

## Predecimos los datos de train y test
predictions = model.predict(data)

predictions = round(pd.DataFrame({"prediction":predictions[:,0] * 100}).astype("float64"), 3)
results_dat = pd.concat([fixed, predictions], axis=1)

## Recuperamos los equipos de la key
results_dat["team1"] = results_dat['match_up'].str[-7:].str.split("@", expand=True)[0]
results_dat["team2"] = results_dat['match_up'].str[-7:].str.split("@", expand=True)[1]


## Agrupamos las predicciones por match_up y seleccionamos a que partidos apostar
final_results = pd.DataFrame()
black_list = []
for i in range(0, results_dat.shape[0]):
    team = results_dat.iloc[i, :].team1
    date = results_dat.iloc[i, :].date
    index = results_dat.loc[(results_dat.team2==team) & (results_dat.date==date)].index
    temp = pd.concat([results_dat.iloc[i, :], results_dat.iloc[index, :].prediction.reset_index(drop=True)], axis=0)
    if temp.team2 in black_list:
        continue
    else:
        black_list.append(team)
        final_results = pd.concat([final_results, temp], axis=1)
        
final_results = final_results.transpose().reset_index(drop=True)
final_results.columns = ["match_up", "date", "prediction_team1", "team1", "team2", "prediction_team2"]
final_results["diff"] = (abs(final_results.prediction_team1 - final_results.prediction_team2)).astype("float64")

## Guardamos las métricas obtenidas
final_results.to_csv(path+"Black_Mamba_Classifier_Pred_Future_games.csv", sep=";", header=True, index=False, decimal=",")
final_results


# ## Betting Odds Scrapping

## Loading urls data
urls = pd.read_csv(path+"url_nba_teams.csv", sep=";")

## Fixing the date
# date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
# date = datetime.now().strftime('%Y-%m-%d')
date = '2021-10-20'

year = date[2:4]
month = date[5:7]
day = date[8:10]

## Generamos el dia para formatear el scrapping
date_flag = day + "/" + month + "/" +  year

## Obtain the predicted games for today
games = pd.read_csv(path+"Black_Mamba_Classifier_Pred_Future_games.csv", sep=";")
games = games.loc[games.date>=date]

## Path para acceder a chromedriver
path_to_chromedriver = path+'chromedriver.exe' 

## Chrome automatizado
ua = UserAgent()
userAgent = ua.random 

options = Options() 
options.add_argument(f'user-agent={userAgent}') 

browser = webdriver.Chrome(executable_path=path_to_chromedriver)


## Url of the betting web
url = "https://s5.sir.sportradar.com/bet365/es/2/season/79153/fixtures/month/{}".format(date[0:7])

## Get daily nba games
try:
    browser.get(url)
    sleep(5)
except:
    sleep(5)
    browser.get(url)
    sleep(5)

try:
    sleep(5)
    browser.find_element_by_xpath('//*[@id="sr-container"]/div/div/div[4]/div/div/div/div/div[2]/div/div/div[2]/button').click()
except:
    sleep(5)
    
## Scrapping odds
table = browser.find_element_by_xpath('//*[@id="sr-container"]/div/div/div[4]/div/div/div/div/div[2]/div/div/div/table/tbody')
table = table.text.split(date_flag)[1].split("\n")[3:]
table = ";".join(table).split("- - - - - ")

all_games = pd.DataFrame()
for line in range(0, len(table)):
    if len(table[line].split(";"))==8:
        game = pd.DataFrame({"teams":[table[line].split(";")[2], table[line].split(";")[4]],
                             "odds":[table[line].split(";")[5], table[line].split(";")[6]]})
        all_games = pd.concat([all_games, game], axis=0)
        
## Merge the data
all_games = all_games.merge(urls.drop(["url", "section"], axis=1), left_on="teams", right_on="name").drop(["team", "teams", "name"], axis=1)
final_results = final_results.merge(all_games, left_on="team1", right_on="codigo", how="left")
final_results = final_results.merge(all_games, left_on="team2", right_on="codigo", how="left")
final_results = final_results.drop(["codigo_x", "codigo_y", "matches_x", "matches_y"], axis=1).dropna(axis=0)

## Format the data
final_results.columns = ["match_up", "date", "pred_team1", "team1", "team2", "pred_team2", "diff", "odds_team1", "odds_team2"]
final_results = final_results[["match_up", "date", "team1", "team2", "pred_team1", "pred_team2", "diff", "odds_team1", "odds_team2"]]

final_results.pred_team1 = final_results.pred_team1.astype("float64")
final_results.pred_team1 = final_results.pred_team1.astype("float64")

final_results.pred_team2 = final_results.pred_team2.astype("float64")
final_results.pred_team2 = final_results.pred_team2.astype("float64")


final_results.odds_team1 = final_results.odds_team1.astype("float64")
final_results.odds_team2 = final_results.odds_team2.astype("float64")

## Obtaining the expected probabilities
final_results["odds_prob1"] = round(100 / final_results.odds_team1, 3)
final_results["odds_prob2"] = round(100 / final_results.odds_team2, 3)

## Obtaining overround and real odds
final_results["overround"] = final_results.odds_prob1 + final_results.odds_prob2 - 100
final_results["real_odds_prob1"] = round(100 / (final_results.odds_team1 + final_results.odds_team1 * final_results.overround / 100), 3)
final_results["real_odds_prob2"] = round(100 / (final_results.odds_team2 + final_results.odds_team2 * final_results.overround / 100), 3)

## Obtaining the Expected Value
final_results["EV%_1"] = (100 / final_results.real_odds_prob1 * final_results.pred_team1 - 100).astype("float64").round(3)
final_results["EV%_2"] = (100 / final_results.real_odds_prob2 * final_results.pred_team2 - 100).astype("float64").round(3)

final_results = final_results.reset_index(drop=True)


# ## Betting Rules

## Url of the standings web
url = "https://stats.nba.com/standings/" 

## Get standings of teams
try:
    browser.get(url)
    sleep(5)
except:
    sleep(5)
    browser.get(url)
    sleep(5)
    
## Accept conditions to continue
try:
    browser.find_element_by_xpath('//*[@id="onetrust-accept-btn-handler"]').click()
    sleep(5)
except:
    sleep(5)
    browser.find_element_by_xpath('//*[@id="onetrust-accept-btn-handler"]').click()
    sleep(5)
    
## East standings
try:
    table = browser.find_element_by_xpath('//*[@id="__next"]/div[2]/div[3]/section/div/div[3]/div[1]/div/div/table/tbody')
    sleep(5)
except:
    sleep(5)
    table = browser.find_element_by_xpath('//*[@id="__next"]/div[2]/div[3]/section/div/div[3]/div[1]/div/div/table/tbody')
    sleep(5)
table = table.text.split("\n")

east = pd.DataFrame()
for i in range(1, len(table)):
    temp = table[i].split(" - ")[0]
    if temp in list(urls.team):
        temp2 = table[i+1].split(" ")[:3]
        temp3 = pd.DataFrame({"Team":temp, "Win":temp2[0], "Lost":temp2[1]}, index=[0])
        east = pd.concat([east, temp3], axis=0).reset_index(drop=True)

east["Total"] = east.Win.astype("int64") + east.Lost.astype("int64")
east["Position"] = list(range(1, east.shape[0]+1))

east = east.merge(urls.drop(["url", "section"], axis=1), left_on="Team", right_on="team").drop(["Team", "team", "name", "matches"], axis=1)


## West standings
try: 
    table = browser.find_element_by_xpath('//*[@id="__next"]/div[2]/div[3]/section/div/div[3]/div[2]/div/div/table/tbody')
    sleep(5)
except:
    sleep(5)
    table = browser.find_element_by_xpath('//*[@id="__next"]/div[2]/div[3]/section/div/div[3]/div[2]/div/div/table/tbody')
    sleep(5)
table = table.text.split("\n")

west = pd.DataFrame()
for i in range(1, len(table)):
    temp = table[i].split(" - ")[0]
    if temp in list(urls.team):
        temp2 = table[i+1].split(" ")[:3]
        temp3 = pd.DataFrame({"Team":temp, "Win":temp2[0], "Lost":temp2[1]}, index=[0])
        west = pd.concat([west, temp3], axis=0).reset_index(drop=True)

west["Total"] = west.Win.astype("int64") + west.Lost.astype("int64")
west["Position"] = list(range(1, west.shape[0]+1))

west = west.merge(urls.drop(["url", "section"], axis=1), left_on="Team", right_on="team").drop(["Team", "team", "name", "matches"], axis=1)


## Total games on the season
total_games = 72

## Comprobamos si los team 1 estan en playoff
for i in range(0, final_results.shape[0]):
    team = final_results.team1[i]
    if team in list(west.codigo):
        conference = west
    elif team in list(east.codigo):
        conference = east
        
    rank = conference.loc[conference.codigo==team, "Position"]
    if int(rank) <= 8: 
        new_wins = int(conference.loc[8].Win) + total_games - conference.loc[8].Total
        
        if int(conference.loc[conference.codigo==team].Win) > new_wins:
            final_results.loc[i, "team1_Playoff"] = 1
        elif int(conference.loc[conference.codigo==team].Win) <= new_wins:
            final_results.loc[i, "team1_Playoff"] = 0
    else:
        final_results.loc[i, "team1_Playoff"] = 0

## Comprobamos si los team 2 estan en playoff
for i in range(0, final_results.shape[0]):
    team = final_results.team2[i]
    if team in list(west.codigo):
        conference = west
    elif team in list(east.codigo):
        conference = east
        
    rank = conference.loc[conference.codigo==team, "Position"]
    if int(rank) <= 8: 
        new_wins = int(conference.loc[8, "Win"]) + total_games - conference.loc[8].Total
        
        if int(conference.loc[i, "Win"]) > new_wins:
            final_results.loc[i, "team2_Playoff"] = 1
        elif int(conference.loc[i, "Win"]) <= new_wins:
            final_results.loc[i, "team2_Playoff"] = 0
    else:
        final_results.loc[i, "team2_Playoff"] = 0

final_results = final_results.fillna(0) 


## Applying rules to bet
final_results["betting"] = "-"

## Range probability rule
final_results.loc[(((final_results["pred_team1"] > 40) & (final_results["pred_team1"] < 60)) |
                   ((final_results["pred_team2"] > 40) & (final_results["pred_team2"] < 60))), "betting"] = "Not bet: Out of range"

## Equal probability rule
final_results.loc[((((final_results["pred_team1"] < 50) & (final_results["pred_team2"] < 50)) |
                    ((final_results["pred_team1"] > 50) & (final_results["pred_team2"] > 50)))), "betting"] = "Not bet: Equal pred"

## Overround rule
final_results.loc[((final_results["overround"] >= 5) &
                   (final_results["betting"] != "Not bet: Out of range") &
                   (final_results["betting"] != "Not bet: Equal pred")), "betting"] = "Not bet: Bad overround"

## Difference probability rule
final_results.loc[((final_results["diff"] <= 10) &
                   (final_results["betting"] != "Not bet: Out of range") & 
                   (final_results["betting"] != "Not bet: Equal pred") &
                   (final_results["betting"] != "Not bet: Bad overround")), "betting"] = "Not bet: Low diff"

## Low Expected Value
final_results.loc[((final_results["betting"] != "Not bet: Out of range") & 
                   (final_results["betting"] != "Not bet: Equal pred") &
                   (final_results["betting"] != "Not bet: Bad overround") &
                   (final_results["betting"] != "Not bet: Low diff") &
                   (final_results["EV%_1"] < 5) &
                   (final_results["EV%_2"] < 5)), "betting"] = "Not bet: Low EV%"

## Betting decision
final_results.loc[((final_results["betting"] == "-") &
                   (final_results["EV%_1"] > final_results["EV%_2"]) &
                   (final_results["EV%_1"] > 10)), "betting"] = "Bet!: Team 1"

final_results.loc[((final_results["betting"] == "-") &
                   (final_results["EV%_2"] > final_results["EV%_1"]) &
                   (final_results["EV%_2"] > 10)), "betting"] = "Bet!: Team 2"

## Too high odd
final_results.loc[(((final_results["betting"] == "Bet!: Team 1") & (final_results["odds_team1"] > 6.66)) |
                   ((final_results["betting"] == "Bet!: Team 2") & (final_results["odds_team2"] > 6.66))), "betting"] = "Not bet: Too high odd"
                   
## Too low odd
final_results.loc[(((final_results["betting"] == "Bet!: Team 1") & (final_results["odds_team1"] < 1.5)) |
                   ((final_results["betting"] == "Bet!: Team 2") & (final_results["odds_team2"] < 1.5))), "betting"] = "Not bet: Too low odd"

# Too high Expected Value
final_results.loc[(((final_results["betting"] == "Bet!: Team 1") & (final_results["EV%_1"] > 100)) |
                   ((final_results["betting"] == "Bet!: Team 2") & (final_results["EV%_2"] > 100))), "betting"] = "Not bet: High EV%"
 
## Play Off ranked
final_results.loc[(((final_results["betting"] == "Bet!: Team 1") & (final_results["team1_Playoff"] == 1)) |
                   ((final_results["betting"] == "Bet!: Team 2") & (final_results["team2_Playoff"] == 1))), "betting"] = "Not bet: Play off ranked"

final_results = final_results.drop(["team1_Playoff", "team2_Playoff"], axis=1)

## Positive Value Range
final_results["odd_limit"] = "-"
final_results.loc[(final_results["betting"] == "Bet!: Team 1"), "odd_limit"] = (105 / final_results.loc[(final_results["betting"] == "Bet!: Team 1"), "pred_team1"]).round(3)
final_results.loc[(final_results["betting"] == "Bet!: Team 2"), "odd_limit"] = (105 / final_results.loc[(final_results["betting"] == "Bet!: Team 2"), "pred_team2"]).round(3)

final_results


## Guardamos los picks
picks = final_results.loc[(final_results["betting"]== "Bet!: Team 1") | (final_results["betting"]== "Bet!: Team 2")]

for i in range(0, len(picks)):
    if picks.iloc[i, -2] == "Bet!: Team 1":
        pick = pd.DataFrame({"Col.1":[picks.iloc[i].team1, "Model Prob. ", "Odd Prob. ", "Expected value ", "Odd limit "],
                             "Col.2":["", picks.iloc[i].pred_team1, picks.iloc[i].odds_prob1, picks.iloc[i].loc["EV%_1"], picks.iloc[i].odd_limit]})
        
    elif picks.iloc[i, -2] == "Bet!: Team 2":
        pick = pd.DataFrame({"Col.1":[picks.iloc[i].team2, "Model Prob. ", "Odd Prob. ", "Expected value ", "Odd limit "],
                             "Col.2":["", picks.iloc[i].pred_team2, picks.iloc[i].odds_prob2, picks.iloc[i].loc["EV%_2"], picks.iloc[i].odd_limit]})
    
    pick.to_csv(path + "daily_pred/picks/{}.csv".format(picks.iloc[i].match_up), sep=";", header=True, index=False)


## Saving results
final_results.to_csv(path+"Black_Mamba_Classifier_Bettting_Future_games.csv", sep=";", header=True, index=False, decimal=",")

## Saving historic results
historic = pd.read_csv(path+"Black_Mamba_Classifier_Historic_Bettting.csv", sep=";", header=0)
final_results = pd.read_csv(path+"Black_Mamba_Classifier_Bettting_Future_games.csv", sep=";", header=0)

historic = pd.concat([historic, final_results], axis=0)
historic = historic.drop_duplicates(subset=['match_up'])
historic.to_csv(path+"Black_Mamba_Classifier_Historic_Bettting.csv", sep=";", header=True, index=False, decimal=",")

## Guardamos el flag de la ejecucion
flag = pd.DataFrame({'flag':1}, index=[0])
flag.to_csv(path+'/scripts/prod/flags/flag_Black_Mamba_Classifier_Predictor.csv', header=True, index=False)
