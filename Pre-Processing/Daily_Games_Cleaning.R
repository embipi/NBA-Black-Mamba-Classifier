
###############################################################################################
###################################### DATA CLEANING NBA ###################################### 
###############################################################################################

####################### Magic Variables ####################### 

## Traza de ejecucion
time <- proc.time()

## Cargamos librerias
library(lubridate)
library(data.table)
library(doParallel)
library(caret)
library(zoo)
library(stringr)

## Indicamos los paths
path = "C:/Users/mibra/Desktop/NBA/"
path_scores = "C:/Users/mibra/Desktop/NBA/daily_pred"
path_stats = "C:/Users/mibra/Desktop/NBA/teams/stats/2020-21/"

## Cargamos las funciones auxiliares
source(file.path(path, "/scripts/modeling/aux_nba.R"))

## Fijamos variables
fixed_var <- c("match_up", "date")

date <- Sys.time()
year <- substr(date, 1, 4)
month <- substr(date, 6, 7)
day <- substr(date, 9, 10)
hour <- as.numeric(substr(date, 12, 13))

if (hour >= 19 & hour < 23){
  hour <- 19
  
} else if (hour == 23 | hour < 2){
  hour <- 23
  
} else if (hour >= 2 & hour < 19){
  hour <- 2
}


####################### Data Loading ####################### 

## Cargamos el fichero basico
nba <- fread(file.path(path, "url_nba_teams.csv"))
nba <- nba[order(team)]

## Creamos un vector con los ficheros del scrapping
scoring_dat <-  fread(file.path(path_scores, "future_games.csv"))
files_stats <- list.files(path = path_stats)

## Leemos todos los ficheros
stats_dat <- data.table()
for (i in seq(1, length(files_stats))){
  ## Stats data
  stats <- fread(file.path(path_stats, files_stats[i]))
  stats$codigo <- nba[i]$codigo
  stats$birth_day <- as.character(stats$birth_day)
  stats$birth_year <- as.character(stats$birth_year)
  stats$min_total <- stats$gp * stats$min
  stats <- stats[order(-min_total), -"min_total"]
  stats_dat <- rbind(stats_dat, stats)
}

## Cargamos el fichero de lesionados y renombramos las variables
if (hour == 19){
  injuries_dat <- fread(file.path(path_scores, paste0("Injury-Report_", year, "-", month, "-", day, "_01PM.csv")))
  
} else if (hour == 23){
  injuries_dat <- fread(file.path(path_scores, paste0("Injury-Report_", year, "-", month, "-", day, "_05PM.csv")))
  
} else if (hour == 2){
  injuries_dat <- fread(file.path(path_scores, paste0("Injury-Report_", year, "-", month, "-", day, "_08PM.csv")))
}

## Damos formato a las columnas
colnames(injuries_dat) <- c("date", "hour", "match_up", "team", "players", "status", "reason")
# injuries_dat <- injuries_dat[3:nrow(injuries_dat)]


####################### Main Cleaning ####################### 

## Rellenamos los huecos vacios
injuries_dat[injuries_dat==""] <- NA
injuries_dat[is.na(status)]$status <- "Not_Submmited"

## Parseamos el csv
for (line in seq(1, nrow(injuries_dat))){
  if (TRUE %in% grepl(pattern="Injury Report", injuries_dat[line])){
    injuries_dat <- injuries_dat[-c(line-1, line, line+1)]
  }
}
  
for (line in seq(1, nrow(injuries_dat))){
  
  if (injuries_dat[line]$players %in% nba$team){
    injuries_dat[line]$team <- injuries_dat[line]$players
    injuries_dat[line]$players <- NA
  }
  
  if (!injuries_dat[line]$team %in% nba$team & !is.na(injuries_dat[line]$team)){
    injuries_dat[line]$players <- injuries_dat[line]$team
    injuries_dat[line]$team <- NA
  }
  
  if (injuries_dat[line]$match_up %in% nba$team){
    injuries_dat[line]$team <- injuries_dat[line]$match_up
    injuries_dat[line]$match_up <- NA
  }

  if (grepl(pattern="/", injuries_dat[line]$date) == is.na(injuries_dat[line]$date) & (grepl("@", injuries_dat[line]$date))){
    injuries_dat[line]$match_up <- injuries_dat[line]$date
    injuries_dat[line]$date <- NA
  }
  
  if (grepl(pattern="/", injuries_dat[line]$date) == is.na(injuries_dat[line]$date) & injuries_dat[line]$date %in% nba$team){
    injuries_dat[line]$team <- injuries_dat[line]$date
    injuries_dat[line]$date <- NA
  }
  
  if (grepl(pattern="/", injuries_dat[line]$date) == is.na(injuries_dat[line]$date)){
    injuries_dat[line]$players <- injuries_dat[line]$date
    injuries_dat[line]$date <- NA
  }

}

## Arreglamos el formato de las varibles
injuries_dat[ , team := na.locf(na.locf(team, na.rm=FALSE), fromLast=TRUE)]
injuries_dat[ , match_up := na.locf(na.locf(match_up, na.rm=FALSE), fromLast=TRUE)]
injuries_dat[ , date := na.locf(na.locf(date, na.rm=FALSE), fromLast=TRUE)]
injuries_dat <- injuries_dat[!is.na(players)]

## Eliminamos los math_up sin informe
injuries_dat <- injuries_dat[!is.na(players) & reason!="NOT YET SUBMITTED" & substr(injuries_dat$date, 4, 5) == day, -c("hour", "reason")]
injuries_dat[status %in% c("Available", "Probable", "Not_Submmited")]$players <- NA

## Eliminamos listado de equipos  sin injuries
my_list <- unique(injuries_dat$match_up)
for (match in my_list){
  temp <- data.table("matchs" =strsplit(match, "@")[[1]])
  temp <- merge(temp, nba, by.x="matchs", by.y="codigo")
  temp <- cbind(temp, match)
  if (FALSE %in% (temp$team %in% injuries_dat[match_up==match]$team)){
    injuries_dat <- injuries_dat[! match_up %in% temp$match]
  }
}

## Reconstruimos el nombre y eliminamos los que no esten lesionados
injuries_dat$name <- sapply(str_split(injuries_dat$players, ","), '[', 2)
injuries_dat$name <- gsub(" ", "", injuries_dat$name)
injuries_dat$surname <- sapply(str_split(injuries_dat$players, ","), '[', 1)
injuries_dat$players <-  paste(injuries_dat$name, injuries_dat$surname, sep=" ")
injuries_dat <- injuries_dat[status!="Not_Submmited", -c("name", "surname")]

## Transformamos la fecha
injuries_dat$date <- as.character(mdy(injuries_dat$date))

## Obtenemos el codigo de cada equipo y agrupamos por fecha
injuries_dat <- merge(injuries_dat, nba[, c("team", "codigo")], by="team")
injuries_dat <- injuries_dat[, .(injuries= paste(unlist(list(players)), collapse=", ")), by=c("date", "codigo")]

## Convertimos la fecha y la desfragmentamos
scoring_dat$date <- as.character(mdy(scoring_dat$date))
scoring_dat$year <- as.character(year(scoring_dat$date))
scoring_dat$month <- as.character(month(scoring_dat$date))
scoring_dat$weekday <- weekdays(as.Date(scoring_dat$date))

## Rellenamos los lesionados de cada equipo
scoring_dat <- merge(scoring_dat, injuries_dat, by.x=c("date", "team1"), by.y=c("date", "codigo"), all.y=T)
setnames(scoring_dat, "injuries", "team1_injuries")
scoring_dat <- merge(scoring_dat, injuries_dat, by.x=c("date", "team2"), by.y=c("date", "codigo"), all.y=T)
setnames(scoring_dat, "injuries", "team2_injuries")

## Eliminamos los registros donde no haya lesionados
scoring_dat[is.na(scoring_dat)] <- "NULL"
scoring_dat <- scoring_dat[date==as.character(Sys.Date()) | date==as.character(Sys.Date()-1)]


## Eliminamos los lesionados en cada partido
stack_1 <- data.table()
stack_2 <- data.table()
for (i in seq(1, nrow(scoring_dat))){
  
  ## Traza
  print(sprintf("Loop %s/%s", i, nrow(scoring_dat)))
  
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
  
  ## Generamos el vector 1 y lo aÃ±adimos al dataframe
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
dat$flag <- 1

## Liberamos memoria
rm(temp, vec1, vec2, out_player1, out_player2, stack_1, stack_2, stats, files_stats, injuries_dat)
gc()

## Cargamos los datos limpios
data <- fread(file.path(path, "clean_data.csv"))
data <- data[, -c("w/l", "target", "rest_days")]
data$flag <- 0
data$flag <- as.numeric(data$flag)

## Reordenamos las columnas para juntar ambos set de datos
data <- data[, colnames(dat), with=F]
data <- rbind(dat, data)
data <- data[, colnames(dat), with=F]

## Obtenemos los dias de descanso entre partido
data$date <- as.Date(data$date)
data[order(date) , rest_days := as.numeric(abs(date - shift(date))), by="team1"]
data[is.na(rest_days)]$rest_days <- 0
data$date <- as.character(data$date)

if (sum(is.na(data))==0){
  ## Guardamos los datos codificados
  write.table(data, file.path(path, "clean_future_games.csv"), row.names=F, sep=";")
  print("Clean Future Games Saved")
}


####################### Main Pre-Processing ####################### 

###### One Hot Encoding ###### 

## Separamos por tipo de variable
categorical_var <- setdiff(colnames(data)[grepl("character", sapply(data, class))], fixed_var)
numerical_var <- setdiff(colnames(data)[grepl("numeric", sapply(data, class))], fixed_var)

## Aplicamos One Hot Encoding
data <- one_hot_encoding(data, categorical_var, numerical_var, fixed_var)

## Filtramos los datos que no queremos predecir
data <- data[flag==1, -"flag"]


###### Important Variables ###### 

## Cargamos las variables importantes
important_variables <- fread(file.path(path, "important_variables.csv"))

## Separamos las variables fijadas y filtramos las variables
fixed <- data[ ,intersect(colnames(data), fixed_var), with=F]
data <- data[, colnames(data) %in% important_variables$Variables, with=F]


###### Tipify ###### 

## Cargamos los vectores de media y desviaciónn típica
mean_vector <- readRDS(file.path(path, "mean_vector.rds"))
sd_vector <- readRDS(file.path(path, "sd_vector.rds"))

## Filtramos las variables utilizadas en el train
mean_vector <- mean_vector[names(mean_vector) %in% intersect(names(mean_vector), important_variables$Variables)]
sd_vector <- sd_vector[names(sd_vector) %in% intersect(names(sd_vector), important_variables$Variables)];

## Nos aseguramos de que mantenga el mismo orden que el set de datos
mean_vector <- mean_vector[colnames(data)]
sd_vector <- sd_vector[colnames(data)]

## Tipificamos los datos
res <- tipify(data, mean_vector, sd_vector)
data <- res$dat

## Juntamos las variables tipificadas y las fijadas
data <- cbind(fixed, data)
data <- data[, c(fixed_var, important_variables$Variables), with=F]

## Liberamos memoria
rm(fixed, res)
gc()

if (sum(is.na(data))==0){
  ## Guardamos los datos codificados
  write.table(data, file.path(path, "final_future_games.csv"), row.names=F, sep=";")
  print("Final Future Games Saved")
}

## Guardamos el flag de la ejecucion
flag <- data.table('flag'=1)
write.table(flag, file.path(path, "scripts/prod/flags/flag_Daily_Games_Cleaning.csv"), row.names=F)
    
## Traza de ejecucion
final_time <- round((proc.time() - time)/60,2)
print(sprintf("Tiempo de ejecucion: %s min.", final_time[3]))
