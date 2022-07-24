#!/usr/bin/env python
# coding: utf-8

## Librerías Básicas
import pandas as pd
import numpy as np
from functools import reduce

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

## Path para acceder a chromedriver
path_to_chromedriver = path+'chromedriver.exe' 
browser = webdriver.Chrome(executable_path=path_to_chromedriver)

t0 = time.time()
url = 'https://stats.nba.com/teams'
temporada = '2019-20'
season = 2

for section in range(1, 7):
    for team in range(1, 6):

        ## Get nba team url
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
        
        ## Select team
        try:
            browser.find_element_by_xpath('/html/body/main/div/div/div[2]/div/div/div[2]/section[2]/div/div/div[2]/section[{}]/div[2]/table/tbody/tr[{}]'.format(section, team)).click()
            sleep(5)
        except:
            sleep(5)
            browser.find_element_by_xpath('/html/body/main/div/div/div[2]/div/div/div[2]/section[2]/div/div/div[2]/section[{}]/div[2]/table/tbody/tr[{}]'.format(section, team)).click()
            sleep(5)
        
        ## Save team url
        new_url_team = browser.current_url
        
        ## Unfold season
        try:
            browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/div[1]/div[1]/div/div/label/select/option[{}]'.format(season)).click()
            sleep(5)                   
        except:
            sleep(5)
            browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/div[1]/div[1]/div/div/label/select/option[{}]'.format(season)).click()
            sleep(5)
            
        ######################## Obtain Roster table ########################
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
        column_names[0] = 'PLAYERS'
        column_names.insert(4,"METRIC")
        column_names.insert(6,"YEAR")

        player_names = []
        player_stats = []
        for i in range(1, len(table)):
            if i%2 != 0:
                player_names.append(table[i])
            else:
                player_stats.append(table[i].split(" "))

        for row in player_stats:
            if len(row) < 11:
                row.append("Empty")
            if len(row) == 11:
                row.append(" ")

        column_names = table[0].split(" ")
        column_names[0] = 'PLAYERS'
        column_names[5] = 'BIRTH_MONTH'
        column_names.insert(5,"METRIC")
        column_names.insert(7,"BIRTH_DAY")
        column_names.insert(8,"BIRTH_YEAR")
        column_names.insert(12,"SCHOOL2")

        ## Creating dataframe
        roster_dat = pd.DataFrame({'players': player_names,
                                   'no.': [i[0] for i in player_stats],
                                   'pos': [i[1] for i in player_stats],
                                   'height': [i[2] for i in player_stats],
                                   'weight': [i[3] for i in player_stats], 
                                   'metric': [i[4] for i in player_stats], 
                                   'birth_month': [i[5] for i in player_stats],
                                   'birth_day': [i[6] for i in player_stats],
                                   'birth_year': [i[7] for i in player_stats],
                                   'age': [i[8] for i in player_stats],
                                   'exp': [i[9] for i in player_stats],
                                   'school': [i[10] for i in player_stats],
                                   'school2': [i[11] for i in player_stats]
                                   }, columns=[i.lower() for i in column_names])

        roster_dat.school = (roster_dat.school + " " + roster_dat.school2)
        school_list = []
        for school in roster_dat.school:
            school_list.append(school.strip())
        roster_dat.school = school_list

        height_list = []
        for height in roster_dat.height:
            height_list.append(height.replace("-", "."))
        roster_dat.height = height_list

        birth_day_list = []
        for day in roster_dat.birth_day:
            birth_day_list.append(day.replace(",", ""))
        roster_dat.birth_day = birth_day_list

        exp_list = []
        for expe in roster_dat.exp:
            if expe == 'R':
                exp_list.append(0)
            else:
                exp_list.append(expe)
        roster_dat.exp = exp_list

        roster_dat = roster_dat.drop(["no.", "metric", "school2"], axis=1)

        roster_dat.height = roster_dat.height.astype("float32")
        roster_dat.weight = roster_dat.weight.astype("float32")
        roster_dat.age = roster_dat.age.astype("float32")
        roster_dat.exp = roster_dat.exp.astype("float32")

        print("\nTamaño de roster:", roster_dat.shape)


        ## Get traditional stats
        try:
            browser.get(new_url_team+"players-traditional/")
            sleep(5)
        except:
            sleep(5)
            browser.get(new_url_team+"players-traditional/")
            sleep(5)   
            
        ## Unfold season
        try:
            browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/div[1]/div[1]/div/div/label/select/option[{}]'.format(season)).click()
            sleep(5)                   
        except:
            sleep(5)
            browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/div[1]/div[1]/div/div/label/select/option[{}]'.format(season)).click()
            sleep(5)

        ######################## Obtain Traditional table ########################
        try:
            table = browser.find_element_by_xpath("/html/body/main/div/div/div/div[4]/div/div/div/div/nba-stat-table[2]/div[2]/div[1]")
            sleep(5)
        except:
            sleep(5)
            table = browser.find_element_by_xpath("/html/body/main/div/div/div/div[4]/div/div/div/div/nba-stat-table[2]/div[2]/div[1]")
            sleep(5)
        table = table.text.split('\n')

        ## Parsing data
        column_names = table[0].split(" ")
        player_names = []
        player_stats = []
        for i in range(1, len(table)):
            if i%2 != 0:
                player_names.append(table[i])
            else:
                player_stats.append(table[i].split(" "))

        ## Creating dataframe
        traditional_dat = pd.DataFrame({'players': player_names,
                                         'gp': [i[0] for i in player_stats],
                                         'min': [i[1] for i in player_stats],
                                         'pts': [i[2] for i in player_stats],
                                         'fgm': [i[3] for i in player_stats], 
                                         'fga': [i[4] for i in player_stats],
                                         'fg%': [i[5] for i in player_stats],
                                         '3pm': [i[6] for i in player_stats],
                                         '3pa': [i[7] for i in player_stats],
                                         '3p%': [i[8] for i in player_stats],
                                         'ftm': [i[9] for i in player_stats],
                                         'fta': [i[10] for i in player_stats],
                                         'ft%': [i[11] for i in player_stats],
                                         'oreb': [i[12] for i in player_stats],
                                         'dreb': [i[13] for i in player_stats],
                                         'reb': [i[14] for i in player_stats],
                                         'ast': [i[15] for i in player_stats],
                                         'tov': [i[16] for i in player_stats],
                                         'stl': [i[17] for i in player_stats],
                                         'blk': [i[18] for i in player_stats],
                                         'pf': [i[19] for i in player_stats],
                                         '+/-': [i[20] for i in player_stats]
                                        }, columns=[i.lower() for i in column_names])

        traditional_dat[traditional_dat.columns[1:]] = traditional_dat[traditional_dat.columns[1:]].apply(pd.to_numeric, downcast='float', errors='coerce')

        print("Tamaño de traditional:", traditional_dat.shape)

        
        ## Get advanced stats
        try:
            browser.get(new_url_team+"players-advanced/")
            sleep(5)
        except:
            sleep(5)
            browser.get(new_url_team+"players-advanced/")
            sleep(5)   
            
        ## Unfold season
        try:
            browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/div[1]/div[1]/div/div/label/select/option[{}]'.format(season)).click()
            sleep(5)                   
        except:
            sleep(5)
            browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/div[1]/div[1]/div/div/label/select/option[{}]'.format(season)).click()
            sleep(5) 

        ######################## Obtaining Avdanced table ########################
        try:
            table = browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/nba-stat-table[2]/div[2]/div[1]')
            sleep(5)
        except:
            sleep(5)
            table = browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/nba-stat-table[2]/div[2]/div[1]')
            sleep(5)
        table = table.text.split('\n')

        ## Parsing data
        column_names = table[0].split(" ")
        column_names[8] = 'AST_RATIO'
        column_names.remove("RATIO") 

        player_names = []
        player_stats = []
        for i in range(1, len(table)):
            if i%2 != 0:
                player_names.append(table[i])
            else:
                player_stats.append(table[i].split(" "))

        ## Obtaining dataframe
        advanced_dat = pd.DataFrame({'players': player_names,
                                     'gp': [i[0] for i in player_stats],
                                     'min': [i[1] for i in player_stats],
                                     'offrtg': [i[2] for i in player_stats],
                                     'defrtg': [i[3] for i in player_stats], 
                                     'netrtg': [i[4] for i in player_stats],
                                     'ast%': [i[5] for i in player_stats],
                                     'ast/to': [i[6] for i in player_stats],
                                     'ast_ratio': [i[7] for i in player_stats],
                                     'oreb%': [i[8] for i in player_stats],
                                     'dreb%': [i[9] for i in player_stats],
                                     'reb%': [i[10] for i in player_stats],
                                     'tov%': [i[11] for i in player_stats],
                                     'efg%': [i[12] for i in player_stats],
                                     'ts%': [i[13] for i in player_stats],
                                     'usg%': [i[14] for i in player_stats],
                                     'pace': [i[15] for i in player_stats],
                                     'pie': [i[16] for i in player_stats]
                                     },columns=[i.lower() for i in column_names])

        advanced_dat[advanced_dat.columns[1:]] = advanced_dat[advanced_dat.columns[1:]].apply(pd.to_numeric, downcast='float', errors='coerce')

        print("Tamaño de advanced:", advanced_dat.shape)


        ## Get misc stats
        try:
            browser.get(new_url_team+"players-misc/")
            sleep(5)
        except:
            sleep(5)
            browser.get(new_url_team+"players-misc/")
            sleep(5)   
            
        ## Unfold season
        try:
            browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/div[1]/div[1]/div/div/label/select/option[{}]'.format(season)).click()
            sleep(5)                   
        except:
            sleep(5)
            browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/div[1]/div[1]/div/div/label/select/option[{}]'.format(season)).click()
            sleep(5) 

        ######################## Obtaining Misc table ########################
        try:
            table = browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/nba-stat-table[2]/div[2]/div[1]')
            sleep(5)
        except:
            sleep(5)
            table = browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/nba-stat-table[2]/div[2]/div[1]')
            sleep(5)
        table = table.text.split('\n')
        table = table[7:]

        ## Parsing data
        column_names = column_names = ["PLAYERS", "GP", "MIN", "PTS_OFF_TO", "2ND_PTS", "FBPS", "PITP", 
                        "OPP_PTS_OFF_TO", "OPP_2ND_PTS", "OPP_FBPS", "OPP_PITP"]

        player_names = []
        player_stats = []
        for i in range(0, len(table)):
            if i%2 == 0:
                player_names.append(table[i])
            else:
                player_stats.append(table[i].split(" "))

        ## Obtaining dataframe
        misc_dat = pd.DataFrame({'players': player_names,
                                 'gp': [i[0] for i in player_stats],
                                 'min': [i[1] for i in player_stats],
                                 'pts_off_to': [i[2] for i in player_stats],
                                 '2nd_pts': [i[3] for i in player_stats], 
                                 'fbps': [i[4] for i in player_stats],
                                 'pitp': [i[5] for i in player_stats],
                                 'opp_pts_off_to': [i[6] for i in player_stats],
                                 'opp_2nd_pts': [i[7] for i in player_stats],
                                 'opp_fbps': [i[8] for i in player_stats],
                                 'opp_pitp': [i[9] for i in player_stats]
                                 },columns=[i.lower() for i in column_names])

        misc_dat[misc_dat.columns[1:]] = misc_dat[misc_dat.columns[1:]].apply(pd.to_numeric, downcast='float', errors='coerce')

        print("Tamaño de misc:", misc_dat.shape)


        ## Get scoring stats
        try:
            browser.get(new_url_team+"players-scoring/")
            sleep(5)
        except:
            sleep(5)
            browser.get(new_url_team+"players-scoring/")
            sleep(5)   
            
        ## Unfold season
        try:
            browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/div[1]/div[1]/div/div/label/select/option[{}]'.format(season)).click()
            sleep(5)                   
        except:
            sleep(5)
            browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/div[1]/div[1]/div/div/label/select/option[{}]'.format(season)).click()
            sleep(5)

        ######################## Obtaining Scoring table ########################
        try:
            table = browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/nba-stat-table[2]/div[2]/div[1]')
            sleep(5)
        except:
            sleep(5)
            table = browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/nba-stat-table[2]/div[2]/div[1]')
            sleep(5)
        table = table.text.split('\n')
        table = table[16:]

        ## Parsing data
        column_names = ["PLAYERS", "GP", "MIN", "%FGA_2PT", "FGA_3PT", "%PTS_2PT", "%PTS_2PT_MR", "%PTS_3PT", "%PTS_FBPS", "%PTS_FT",
                        "%PTS_OFFTO", "%PTS_PITP", "2FGM_%AST", "2FGM_%UAST", "3FGM_%AST", "3FGM_%UAST", "FGM_%AST", "FGM_%UAST"]

        player_names = []
        player_stats = []
        for i in range(0, len(table)):
            if i%2 == 0:
                player_names.append(table[i])
            else:
                player_stats.append(table[i].split(" "))

        ## Obtaining dataframe
        scoring_dat = pd.DataFrame({'players': player_names,
                                    'gp': [i[0] for i in player_stats],
                                    'min': [i[1] for i in player_stats],
                                    '%fga_2pt': [i[2] for i in player_stats],
                                    'fga_3pt': [i[3] for i in player_stats], 
                                    '%pts_2pt': [i[4] for i in player_stats],
                                    '%pts_2pt_mr': [i[5] for i in player_stats],
                                    '%pts_3pt': [i[6] for i in player_stats],
                                    '%pts_fbps': [i[7] for i in player_stats],
                                    '%pts_ft': [i[8] for i in player_stats],
                                    '%pts_offto': [i[9] for i in player_stats],
                                    '%pts_pitp': [i[10] for i in player_stats],
                                    '2fgm_%ast': [i[11] for i in player_stats],
                                    '2fgm_%uast': [i[12] for i in player_stats],
                                    '3fgm_%ast': [i[13] for i in player_stats],
                                    '3fgm_%uast': [i[14] for i in player_stats],
                                    'fgm_%ast': [i[15] for i in player_stats],
                                    'fgm_%uast': [i[16] for i in player_stats]
                                    },columns=[i.lower() for i in column_names])

        scoring_dat[scoring_dat.columns[1:]] = scoring_dat[scoring_dat.columns[1:]].apply(pd.to_numeric, downcast='float', errors='coerce')

        print("Tamaño de scoring:", scoring_dat.shape)


        ## Get usage stats
        try:
            browser.get(new_url_team+"players-usage/")
            sleep(5)
        except:
            sleep(5)
            browser.get(new_url_team+"players-usage/")
            sleep(5)   
            
        ## Unfold season
        try:
            browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/div[1]/div[1]/div/div/label/select/option[{}]'.format(season)).click()
            sleep(5)                   
        except:
            sleep(5)
            browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/div[1]/div[1]/div/div/label/select/option[{}]'.format(season)).click()
            sleep(5)

        ######################## Obtaining Usage table ########################
        try:
            table = browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/nba-stat-table[2]/div[2]/div[1]')
            sleep(5)
        except:
            sleep(5)
            table = browser.find_element_by_xpath('/html/body/main/div/div/div/div[4]/div/div/div/div/nba-stat-table[2]/div[2]/div[1]')
            sleep(5)
        table = table.text.split('\n')

        ## Parsing data
        column_names = table[0].split(" ")
        player_names = []
        player_stats = []
        for i in range(1, len(table)):
            if i%2 != 0:
                player_names.append(table[i])
            else:
                player_stats.append(table[i].split(" "))

        ## Obtaining dataframe  
        usage_dat = pd.DataFrame({'players': player_names,
                                  'gp': [i[0] for i in player_stats],
                                  'min': [i[1] for i in player_stats],
                                  'usg%': [i[2] for i in player_stats],
                                  '%fgm': [i[3] for i in player_stats], 
                                  '%fga': [i[4] for i in player_stats],
                                  '%3pm': [i[5] for i in player_stats],
                                  '%3pa': [i[6] for i in player_stats],
                                  '%ftm': [i[7] for i in player_stats],
                                  '%fta': [i[8] for i in player_stats],
                                  '%oreb': [i[9] for i in player_stats],
                                  '%dreb': [i[10] for i in player_stats],
                                  '%reb': [i[11] for i in player_stats],
                                  '%ast': [i[12] for i in player_stats],
                                  '%tov': [i[13] for i in player_stats],
                                  '%stl': [i[14] for i in player_stats],
                                  '%blk': [i[15] for i in player_stats],
                                  '%blka': [i[16] for i in player_stats],
                                  '%pf': [i[17] for i in player_stats],
                                  '%pfd': [i[18] for i in player_stats],
                                  '%pts': [i[19] for i in player_stats]
                                  },columns=[i.lower() for i in column_names])

        usage_dat[usage_dat.columns[1:]] = usage_dat[usage_dat.columns[1:]].apply(pd.to_numeric, downcast='float', errors='coerce')

        print("Tamaño de usage:", usage_dat.shape)

        ######################## Merge All Tables ########################
        data_frames = [traditional_dat,
                       roster_dat,
                       advanced_dat.drop(["gp", "min"], axis=1),
                       misc_dat.drop(["gp", "min"], axis=1),
                       scoring_dat.drop(["gp", "min"], axis=1),
                       usage_dat.drop(["gp", "min", "usg%"], axis=1)]

        stats_dat = reduce(lambda x, y: pd.merge(x, y, on='players', how='inner'), data_frames)

        ## Obtaining name of team to save
        sec = urls.section.unique()[section-1]
        name = urls.loc[urls['section']==sec].team.reset_index(drop=True)[team-1]
        
        print("Tamaño de {} stats: {}".format(name, stats_dat.shape))

        stats_dat.to_csv(path+'/teams/stats/'+temporada+'/'+name+' player stats.csv', sep=";", header=True, index=False)

total_time = round((time.time() - t0)/60, 2)
print("\nEnd Scrapping Player Stats Loop:", total_time)
