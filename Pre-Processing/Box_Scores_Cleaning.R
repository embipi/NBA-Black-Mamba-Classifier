
###############################################################################################
###################################### DATA CLEANING NBA ###################################### 
###############################################################################################


####################### Magic Variables ####################### 

## Traza de ejecución
time <- proc.time()

## Cargamos librerias
library(lubridate)
library(data.table)

## Indicamos los paths
path = "C:/Users/mibra/Desktop/NBA/"
path_scores = "C:/Users/mibra/Desktop/NBA/teams/scores/"
path_stats = "C:/Users/mibra/Desktop/NBA/teams/stats/"

## Cargamos las funciones auxiliares
source(file.path(path, "/scripts/modeling/aux_nba.R"))

## Fijamos variables
fixed_var <- c("match_up", "date", "w/l", "target", "dataset")


####################### Data Loading ####################### 

## Cargamos el fichero básico
nba <- fread(file.path(path, "url_nba_teams.csv"))
nba <- nba[order(team)]

## Generamos un listado de las temporadas
season <- list.files(path = path_scores)

data <- data.table()
for (temporada in seq(1, length(season)-1)){
  
  ## Traza
  print(sprintf("Season %s/%s", temporada, length(season)-1))
  
  ## Creamos un vector con los ficheros del scrapping
  files_scores <- list.files(path = file.path(path_scores, season[temporada]))
  files_stats <- list.files(path = file.path(path_stats, season[temporada]))

  ## Leemos todos los ficheros
  scoring_dat <- data.table()
  stats_dat <- data.table()
  for (i in seq(1, length(files_scores))){
    ## Scoring data
    scoring <- fread(file.path(path_scores, season[temporada], files_scores[i]))
    scoring_dat <- rbind(scoring_dat, scoring)
    
    ## Stats data
    stats <- fread(file.path(path_stats, season[temporada], files_stats[i]) )
    stats$codigo <- nba[i]$codigo
    stats$birth_day <- as.character(stats$birth_day)
    stats$birth_year <- as.character(stats$birth_year)
    stats$min_total <- stats$gp * stats$min
    stats <- stats[order(-min_total), -"min_total"]
    stats_dat <- rbind(stats_dat, stats)
  }
  
  ## liberamos memoria
  rm(stats, scoring, files_scores, files_stats)
  gc()
  
  
  ####################### Main Cleaning ####################### 
  
  ## Corregimos los codigos pasados
  scoring_dat[team1=="NOH" | team1=="NOK"]$team1 <- "NOP"
  scoring_dat[team2=="NOH" | team2=="NOK"]$team2 <- "NOP"
  
  scoring_dat[team1=="NJN"]$team1 <- "BKN"
  scoring_dat[team2=="NJN"]$team2 <- "BKN"
  
  scoring_dat[team1=="SEA"]$team1 <- "OKC"
  scoring_dat[team2=="SEA"]$team2 <- "OKC"
  
  ## Convertimos la fecha y la desfragmentamos
  scoring_dat$date <- mdy(scoring_dat$date)
  scoring_dat$year <- as.character(year(scoring_dat$date))
  scoring_dat$month <- as.character(month(scoring_dat$date))
  scoring_dat$weekday <- weekdays(scoring_dat$date)
  
  ## Obtenemos los dias de descanso entre partido
  scoring_dat[order(date) , rest_days := as.numeric(abs(date - shift(date))), by="team1"]
  scoring_dat[is.na(rest_days)]$rest_days <- 0
  
  ## Eliminamos los lesionados en cada partido
  stack_1 <- data.table()
  stack_2 <- data.table()
  for (i in seq(1, nrow(scoring_dat))){
    
    if (i%%100==0){
      ## Traza
      print(sprintf("Loop %s/%s", i, nrow(scoring_dat)))
    }
    
    ###### LOOP 1 ###### 
    
    ## Lesionados equipo 1  
    t1 <- scoring_dat[i]$team1
    injuries1 <- scoring_dat[i]$team1_injuries
      
    out_player1 <- data.table()
    for (player in stats_dat[codigo==t1]$players){
      if (length(grep(player, injuries1))>0){
        out_player1 <- rbind(out_player1, player)
      }
    }
    
    ## Seleccionamos los jugadores para la tabla final
    temp <- stats_dat[codigo==t1 & !players %in% out_player1$x, -"codigo"][1:5]
    
    ## Generamos el vector 1 y lo añadimos al dataframe
    names1 <- c()
    vec1 <- data.table("try"="NULL")
    for (n in seq(1,5)){
      names1 <- c(names1, paste0(colnames(stats_dat[, -"codigo"]),"_team1_", n))
      vec1 <- cbind(vec1, temp[n, -"codigo"])
      vec1 <- vec1[, -"try"]
      colnames(vec1) <- names1
    }
    stack_1 <- rbind(stack_1, vec1)
    stack_1 <- stack_1[, colnames(vec1), with=F]
    
    ###### LOOP 2 ###### 
    
    ## Lesionados equipo 2  
    t2 <- scoring_dat[i]$team2
    injuries2 <- scoring_dat[i]$team2_injuries
        
    out_player2 <- data.table()
    for (player in stats_dat[codigo==t2]$players){
      if (length(grep(player, injuries2))>0){
        out_player2 <- rbind(out_player2, player)
      }
    }
    ## Seleccionamos los jugadores para la tabla final
    temp <- stats_dat[codigo==t2 & !players %in% out_player2$x][1:5]
    
    ## Generamos el vector 2 y lo añadimos al dataframe
    names2 <- c()
    vec2 <- data.table("try"="NULL")
    for (n in seq(1,5)){
      names2 <- c(names2, paste0(colnames(stats_dat[, -"codigo"]),"_team2_", n))
      vec2 <- cbind(vec2, temp[n, -"codigo"])
      vec2 <- vec2[, -"try"]
      colnames(vec2) <- names2
    }
    stack_2 <- rbind(stack_2, vec2)
    stack_2 <- stack_2[, colnames(vec2), with=F]
  }

  ## Juntamos todas las tablas en una
  dat <- cbind(scoring_dat[, -c("team1_injuries", "team2_injuries")], stack_1, stack_2)
  
  ## Guardamos en una variable los datos de cada temporada
  data <- rbind(data, dat)
  data <- data[, colnames(dat), with=F]
  
  ## Liberamos memoria
  rm(temp, vec1, vec2, out_player1, out_player2, stack_1, stack_2)
  gc()

}


if (sum(is.na(data))==0){
  ## Guardamos los datos codificados
  write.table(data, file.path(path, "clean_data_seasons.csv"), row.names=F, sep=";")
  print("Clean Data Seasons Saved")
}


# Traza de ejecución
final_time <- round((proc.time() - time)/60,2)
print(sprintf("Tiempo de ejecuci�n: %s min.", final_time[3]))


