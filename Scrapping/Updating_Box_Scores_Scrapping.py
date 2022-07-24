#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Librerías Básicas
import pandas as pd
import numpy as np
from datetime import datetime

## Librerías para Warnings
import warnings
warnings.filterwarnings('ignore')

## Librerías para Scrapping
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium import webdriver
import time
from time import sleep


# In[2]:


## Declaramos path y leemos archivo de urls
path = "C:/Users/mibra/Desktop/NBA/"
urls = pd.read_csv(path+"url_nba_teams.csv", header=0, sep=";")

## Path para acceder a chromedriver
path_to_chromedriver = path+'chromedriver.exe' 

## Chrome automatizado
browser = webdriver.Chrome(executable_path=path_to_chromedriver)


# In[3]:


t0 = time.time()

url = 'https://www.nba.com/teams'
temporada = '2021-22'
season = 1

## Get nba ulr and accept conditions to continue
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
    
for team in urls.url:
        
    ## Read scrapped files     
    name = [x for x in urls.loc[urls.url==team].team.values][0]
    temp = pd.read_csv(path+"teams/scores/2020-21/"+ name +" game score.csv", header=0, sep=";", encoding='latin-1')

    length = temp.shape[0]
       
    print("\nTamaño de {} Scrapped Scores & Injuries: {}".format(name, temp.shape))
        
    url = 'https://www.nba.com/stats/team/'+str(team)+'/boxscores-traditional/'

    ## Get nba team url
    try:
        browser.get(url)
        sleep(5)
    except:
        sleep(5)
        browser.get(url)
        sleep(5)

    ## Unfold season
    try:                               
        browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/div[1]/div[1]/div/div/label/select/option[{}]'.format(season)).click()
        sleep(5)                   
    except:
        sleep(5)
        browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/div[1]/div[1]/div/div/label/select/option[{}]'.format(season)).click()
        sleep(5)
            
    ## Select all pages
    try:
        browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/nba-stat-table/div[1]/div/div/select/option[1]').click()
        sleep(5)
    except:
        sleep(5)
#       browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/nba-stat-table/div[1]/div/div/select/option[1]').click()
        sleep(5)
    
    ######################## Obtain Box Scores table ########################
    try:
        table = browser.find_element_by_class_name('nba-stat-table__overflow')
        sleep(5)
    except:
        sleep(5)
        table = browser.find_element_by_class_name('nba-stat-table__overflow')
        sleep(5)
    table = table.text.split('\n')

    ## Parsing data
    column_names = table[0].split(" ")
    column_names[0] = 'MONTH'
    column_names[1] = 'DAY'
    column_names.insert(2, "YEAR")
    column_names.insert(3, "TEMP")
    column_names.insert(4, "TEAM1")
    column_names.insert(5, "VISITOR/HOME")
    column_names.insert(6, "TEAM2")
    column_names[7] = 'W/L'
    column_names[-1] = "TARGET"        
        
    game_stats = []
    for i in range(1, len(table)):
        game_stats.append(table[i].split(" "))

    ## Creating dataframe
    box_scores_dat = pd.DataFrame({'month': [i[0] for i in game_stats],
                                   'day': [i[1] for i in game_stats],
                                   'year': [i[2] for i in game_stats],
                                   'temp': [i[3] for i in game_stats],
                                   'team1': [i[4] for i in game_stats],
                                   'visitor/home': [i[5] for i in game_stats],
                                   'team2': [i[6] for i in game_stats],
                                   'w/l': [i[7] for i in game_stats],
                                   'min': [i[8] for i in game_stats],
                                   'pts': [i[9] for i in game_stats],
                                   'fgm': [i[10] for i in game_stats], 
                                   'fga': [i[11] for i in game_stats],
                                   'fg%': [i[12] for i in game_stats],
                                   '3pm': [i[13] for i in game_stats],
                                   '3pa': [i[14] for i in game_stats],
                                   '3p%': [i[15] for i in game_stats],
                                   'ftm': [i[16] for i in game_stats],
                                   'fta': [i[17] for i in game_stats],
                                   'ft%': [i[18] for i in game_stats],
                                   'oreb': [i[19] for i in game_stats],
                                   'dreb': [i[20] for i in game_stats],
                                   'reb': [i[21] for i in game_stats],
                                   'ast': [i[22] for i in game_stats],
                                   'tov': [i[23] for i in game_stats],
                                   'stl': [i[24] for i in game_stats],
                                   'blk': [i[25] for i in game_stats],
                                   'pf': [i[26] for i in game_stats],
                                   'target': [i[27] for i in game_stats]
                                    }, columns=[i.lower() for i in column_names])

    box_scores_dat["match_up"] = (box_scores_dat.month+" "+ box_scores_dat.day+" "+box_scores_dat.year+" - "+box_scores_dat.team1+"@"+box_scores_dat.team2)
    box_scores_dat["date"] = (box_scores_dat.month+" "+ box_scores_dat.day+" "+box_scores_dat.year)
    box_scores_dat = box_scores_dat.loc[:,["match_up", "date", "team1", "visitor/home", "team2", "w/l", "target"]]

    tag_list = []
    for tag in box_scores_dat["visitor/home"]:
        if tag == 'vs.':
            tag_list.append("home")
        else:
            tag_list.append("visitor")
    box_scores_dat["visitor/home"] = tag_list

    date_list = []
    for date in box_scores_dat.date:
        date_list.append(datetime.strptime(date.replace(",", ""), '%b %d %Y').strftime('%m/%d/%Y'))
    box_scores_dat.date = date_list

    ######################## Obtain New Box Scores table ########################
       
    ## Obtaining number of rows
    try:
        table = browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/nba-stat-table/div[2]/div[2]/table/tbody')
        sleep(5)
    except:
        sleep(5)
        table = browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/nba-stat-table/div[2]/div[2]/table/tbody')
        sleep(5)
            
    scrapping = int(len(table.text.split("\n")))
        
    ## Selecting unscrapped rows
    scrapping = scrapping - length
    box_scores_dat = box_scores_dat.iloc[:scrapping, ]
         
    ## Loop over each match
    injury_dat = pd.DataFrame()
    for game in range(1, scrapping+1):
        print("loop:{}/{}".format(game, scrapping))
        
        ## Select injuries
        try:
            element = WebDriverWait(browser,180).until(EC.element_to_be_clickable((By.XPATH, '/html/body/main/div/div/div/div[4]/div/div/div/div/nba-stat-table/div[2]/div[2]/table/tbody/tr[{}]/td/a'.format(game))))
            browser.execute_script("arguments[0].click();", element)
            sleep(5)                                                          
        except:                                                                           
            sleep(5)
            element = WebDriverWait(browser,180).until(EC.element_to_be_clickable((By.XPATH, '/html/body/main/div/div/div/div[4]/div/div/div/div/nba-stat-table/div[2]/div[2]/table/tbody/tr[{}]/td/a'.format(game))))
            browser.execute_script("arguments[0].click();", element)
            sleep(5)
                
        ## Move to box-score injuries
        new_url_injuries = browser.current_url+"/box-score"
        try:
            sleep(5)
            browser.get(new_url_injuries)
            sleep(5)               
        except:
            sleep(5)
            browser.get(new_url_injuries)
            sleep(5)  
  
        ######################## Obtain Injuries table ########################
            
        ## Parsing data of injuries  
        injury_list = []
        try:
            table = browser.find_element_by_xpath('//*[@id="__next"]/div[2]/div[4]/aside'.format(i))
            sleep(5)
        except:
            sleep(5)
            table = browser.find_element_by_xpath('//*[@id="__next"]/div[2]/div[4]/aside'.format(i))
            sleep(5)
                    
        injuries = table.text.split('\n')
            
        if len(injuries) > 2:
            for i in range(1, 3):
                injuries_parse = injuries[i][5:].split(", ")
                injury_list.append(injuries_parse) 
        else:
            for i in range(1, 2):
                injuries_parse = injuries[i][5:].split(", ")
                injury_list.append(injuries_parse) 
                injury_list.append(["NULL"])
                
        ## Creating dataframe
        if box_scores_dat.loc[game-1, "visitor/home"] == "visitor":
            injury_players = pd.DataFrame({'team1_injuries': ", ".join(injury_list[0]),
                                           'team2_injuries': ", ".join(injury_list[1])}, index=[0])
        elif box_scores_dat.loc[game-1, "visitor/home"] == "home":
            injury_players = pd.DataFrame({'team1_injuries': ", ".join(injury_list[1]),
                                           'team2_injuries': ", ".join(injury_list[0])}, index=[0]) 

        injury_dat = injury_dat.append(injury_players).reset_index(drop=True)

        ## Go back
        try:
            browser.get(url)
            sleep(5)
        except:
            sleep(5)
            browser.get(url)
            sleep(5)
            
        ## Unfold season
        try:
            browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/div[1]/div[1]/div/div/label/select/option[{}]'.format(season)).click()
            sleep(5)                   
        except:
            sleep(5)
            browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/div[1]/div[1]/div/div/label/select/option[{}]'.format(season)).click()
            sleep(5)
        
    ## Concatenating target and injuries
    box_scores_dat = pd.concat([box_scores_dat, injury_dat], axis=1)
    print("Tamaño de {} New Scores & Injuries: {}".format(name, box_scores_dat.shape))
    box_scores_dat.to_csv(path+'/teams/scores/'+temporada+'/update/'+name+' game score update.csv', sep=";", header=True, index=False)
        
    ## Concatenating scrapped data with new data
    box_scores_dat = pd.concat([box_scores_dat, temp], axis=0)
               
    ## Contamos los posibles duplicados y los eliminamos
    duplicated_rows = box_scores_dat.duplicated(subset='match_up').sum()   
    if (duplicated_rows > 0):
        box_scores_dat = box_scores_dat.drop_duplicates(subset='match_up', keep='last').reset_index(drop=True)
        print("Número de filas duplicadas eliminadas:", duplicated_rows)
    else:
        print("No se han encontrado filas duplicadas")
    
    ## Save final data
    print("Tamaño de {} Historic Scores & Injuries: {}".format(name, box_scores_dat.shape))
    box_scores_dat.to_csv(path+'/teams/scores/'+temporada+'/'+name+' game score.csv', sep=";", header=True, index=False)
        
total_time = round((time.time() - t0)/60, 2)
print("\nEnd Scrapping Scoring & Injuries Loop:", total_time)

