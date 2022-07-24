#!/usr/bin/env python
# coding: utf-8

## Librerías Básicas
import pandas as pd
import numpy as np
from datetime import datetime

## Librerías para Warnings
import warnings
warnings.filterwarnings('ignore')

## Librerías para Scrapping
from selenium import webdriver
import time
from time import sleep

## Declaramos path y leemos archivo de urls
path = "C:/Users/mibra/Desktop/NBA/"
urls = pd.read_csv(path+"url_nba_teams.csv", header=0, sep=";")

## Obtenemos la fecha del dia actual
date = time.strftime("%Y-%m-%d")

## Dividimos la fecha en cada elemento de la url
year = date[0:4]
month = date[5:7]
day = date[8:11]

## Generamos un calendario para la variable "match_up"
calendar = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

## Path para acceder a chromedriver
path_to_chromedriver = path+'chromedriver.exe' 

## Chrome automatizado
browser = webdriver.Chrome(executable_path=path_to_chromedriver)

## Format URL
url = 'https://www.nba.com/games?'
   
## Get url and accept conditions to continue
try:
    browser.get(url)
    sleep(5)
    browser.find_element_by_xpath('//*[@id="onetrust-accept-btn-handler"]').click()
    sleep(5)
except:
    sleep(5)
    browser.get(url)
    sleep(5)
    browser.find_element_by_xpath('//*[@id="onetrust-accept-btn-handler"]').click()
    sleep(5)
    
## Eliminate spam   
try: 
    browser.find_element_by_xpath('/html/body/div[4]/div[2]/button').click()
    sleep(5)
except:
    sleep(5)
    browser.find_element_by_xpath('/html/body/div[4]/div[2]/button').click()
    sleep(5)

t0 = time.time()

num_days = 7

matchs = pd.DataFrame()
for days in range(0, num_days):
    
    ## Traza
    print("\nLoop: {}/{}".format(days+1, num_days))
    
    ## Format URL
    if (int(day) <= 9) & (int(day) + days <= 9):
        url = 'https://www.nba.com/games?date='+year+'-'+month+'-'+"0"+str(int(day)+days)
    else:
        url = 'https://www.nba.com/games?date='+year+'-'+month+'-'+str(int(day)+days)
    
    fecha = month+"/"+str(int(day)+days)+"/"+year
    match_up = calendar[int(month)-1]+" "+str(int(day)+days)+", "+year+" - "

    ## Get daily nba games
    try:
        browser.get(url)
        sleep(5)
    except:
        sleep(5)
        browser.get(url)
        sleep(5)
     
    ## Get total games of the day
    try:
        games = len(browser.find_elements_by_class_name('shadow-block')) 
        sleep(5)
    except:
        sleep(5)
        games = len(browser.find_elements_by_class_name('shadow-block')) 
        sleep(5)
   
    ## Get match ups
    for game in range(1, games+1):                 
                                                   
        try:                                       
            table = browser.find_element_by_xpath('//*[@id="__next"]/div[2]/div[3]/div/div[1]/div[1]/div[{}]/div[1]/a/div'.format(str(game+1)))
            sleep(5)                               
        except:
            sleep(5)
            table = browser.find_element_by_xpath('//*[@id="__next"]/div[2]/div[3]/div/div[1]/div[1]/div[{}]/div[1]/a/div'.format(str(game+1)))
            sleep(5)
        table = table.text.split("\n")
        visitor = table[0]
        home = table[-2]
        
        ## Damos formato a los nombres
        visitor = pd.DataFrame({'matches':visitor}, index=[0])
        visitor = urls.merge(visitor, on="matches").codigo[0]

        home = pd.DataFrame({'matches':home}, index=[0])
        home = urls.merge(home, on="matches").codigo[0]

        ## Generamos un registro de visitor y otro de home
        df1 = pd.DataFrame({"match_up": match_up+visitor+"@"+home,
                            "date":fecha,
                            "team1": visitor,
                            "visitor/home": "visitor",
                            "team2": home}, index=[0])

        df2 = pd.DataFrame({"match_up": match_up+home+"@"+visitor,
                            "date":fecha,
                            "team1": home,
                            "visitor/home": "home",
                            "team2": visitor}, index=[0])

        ## Unimos cada registro a la tabla final
        matchs = pd.concat([matchs, df1, df2], axis=0).reset_index(drop=True)
        
## Guardamos los datos a predecir        
print("\nTamaño de Match Ups: {}".format(matchs.shape))

matchs.to_csv(path+'/daily_pred/future_games.csv', sep=";", header=True, index=False)

## Guardamos el flag de la ejecucion
flag = pd.DataFrame({'flag':1}, index=[0])
flag.to_csv(path+'/scripts/prod/flags/flag_Daily_Games_Scrapping.csv', header=True, index=False)

total_time = round((time.time() - t0)/60, 2)
print("\nEnd Scrapping Daily Games Loop:", total_time)
