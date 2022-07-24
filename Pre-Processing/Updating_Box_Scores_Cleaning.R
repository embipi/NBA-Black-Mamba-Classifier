
###############################################################################################
###################################### DATA CLEANING NBA ###################################### 
###############################################################################################


####################### Magic Variables ####################### 

## Traza de ejecución
time <- proc.time()

## Cargamos librerias
library(lubridate)
library(data.table)
library(doParallel)
library(caret)

## Indicamos los paths
path = "C:/Users/mibra/Desktop/NBA/"
path_scores = "C:/Users/mibra/Desktop/NBA/teams/scores/2020-21/"
path_stats = "C:/Users/mibra/Desktop/NBA/teams/stats/2020-21/"

## Cargamos las funciones auxiliares
source(file.path(path, "scripts/modeling/aux_nba.R"))

## Fijamos variables
fixed_var <- c("match_up", "date", "w/l", "target", "dataset")

## Porcentajes de cada muestra
perc_split <- c(0.8, 0.15, 0.05)

## Tipo de divisiÃ³n de la muestra
split_type <- 'temporal'

## Indicamos el numero de variables totales
num_imp_var <- 200


####################### Data Loading ####################### 

## Cargamos el fichero bÃ¡sico
nba <- fread(file.path(path, "url_nba_teams.csv"))
nba <- nba[order(team)]

## Creamos un vector con los ficheros del scrapping
files_scores <- list.files(path = path_scores)
files_scores <- files_scores[grep("csv", files_scores)]
files_stats <- list.files(path = path_stats)

## Leemos todos los ficheros
scoring_dat <- data.table()
stats_dat <- data.table()
for (i in seq(1, length(files_scores))){
  
  ## Scoring data
  scoring <- fread(file.path(path_scores, files_scores[i]))
  scoring_dat <- rbind(scoring_dat, scoring)

  ## Stats data
  stats <- fread(file.path(path_stats, files_stats[i]))
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

## Convertimos la fecha y la desfragmentamos
scoring_dat$date <- mdy(scoring_dat$date)
scoring_dat$year <- as.character(year(scoring_dat$date))
scoring_dat$month <- as.character(month(scoring_dat$date))
scoring_dat$weekday <- weekdays(as.Date(scoring_dat$date))

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
  
  ## Generamos el vector 2 y lo aÃ±adimos al dataframe
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

## Liberamos memoria
rm(temp, vec1, vec2, out_player1, out_player2, stack_1, stack_2, scoring_dat, stats_dat)
gc()

## Cargamos los datos limpios
data <- fread(file.path(path, "clean_data_seasons.csv"))

## Reordenamos las columnas para juntar ambos set de datos
data <- data[, colnames(dat), with=F]
dat$date <- as.character(dat$date)
data <- rbind(dat, data)
data <- data[, colnames(dat), with=F]


if (sum(is.na(data))==0){
  ## Guardamos los datos codificados
  write.table(data, file.path(path, "clean_data.csv"), row.names=F, sep=";")
  print("Clean Data Saved")
}

####################### Main Pre-Processing ####################### 

###### Train/Val Splitting ###### 

if (split_type=="temporal"){
  ## Ordenamos los datos segÃºn fecha de partido
  data <- data[order(date), ]
  
} else if (split_type=="random"){
  ## Reordenamos los datos aleatoriamente
  data <- data[sample(.N, nrow(data)), ]
}

## Dividimos los datos
data$dataset <- "test"
data$index <- 1:nrow(data)
ind_train <- data$index[1:(nrow(data)*perc_split[1])]
ind_val <- data$index[(max(ind_train)+1):(max(ind_train)+1+(nrow(data)*perc_split[2]))]
data[index %in% ind_train]$dataset <- "train"
data[index %in% ind_val]$dataset <- "val"
data <- data[, -"index", with = F]


###### One Hot Encoding ###### 

## Separamos por tipo de variable
categorical_var <- setdiff(colnames(data)[grepl("character", sapply(data, class))], fixed_var)
numerical_var <- setdiff(colnames(data)[grepl("numeric", sapply(data, class))], fixed_var)

## Aplicamos One Hot Encoding
data <- one_hot_encoding(data, categorical_var, numerical_var, fixed_var)


###### Tipify ###### 

## Separamos las variables fijadas
fixed <- data[ ,intersect(colnames(data), fixed_var), with=F]
data <- data[ , -setdiff(fixed_var, "dataset"), with =F]

## Calculamos los vectores de media y desviaciÃ³n tipica
mean_vector <- sapply(data[dataset == "train", -"dataset", with = F], mean)
sd_vector <- sapply(data[dataset == "train", -"dataset", with = F], sd)

## Guardamos los vectores de tipificaciÃ³n
saveRDS(mean_vector, file.path(path, "mean_vector.rds"))
saveRDS(sd_vector, file.path(path, "sd_vector.rds"))

## Tipificamos los datos
res <- tipify(data[, -"dataset", with = F], mean_vector, sd_vector)
data <- res$dat

## Juntamos las variables tipificadas y las fijadas
data <- cbind(fixed, data)

## Liberamos memoria
rm(fixed, res, dat)
gc()


# if (sum(is.na(data))==0){
#   ## Guardamos los datos codificados
#   write.table(data, file.path(path, "encoded_data.csv"), row.names=F, sep=";")
#   print("Encoded Data Saved")
# }
    

###### Remove Redundant ###### 

## Pasamos el target a binario
data[`w/l`=="L"]$`w/l` <- "0"
data[`w/l`=="W"]$`w/l` <- "1"
data[(is.na(`w/l`) | `w/l`=="") & target > 0]$`w/l` <- "1"
data[(is.na(`w/l`) | `w/l`=="") & target < 0]$`w/l` <- "0"
data$`w/l` <- as.integer(data$`w/l`)

## Obtenemos la correlación entre variables
correlations <- abs(cor(data.table(target=data[dataset=="train"]$`w/l`, data[dataset=="train", -fixed_var, with=F])))

## Eliminamos las variables redundantes
redundant_variables <- remove_redundant(correlations, redundant_threshold=0.95)
data <- data[ ,setdiff(colnames(data), redundant_variables), with=F]
print(sprintf("Variables redundantes eliminadas: %s", length(redundant_variables)))


###### Variable Importance ######

## Obtenemos las variables más importantes
important_variables <- select_important(data[dataset=="train"], data[dataset=="train"]$`w/l`)

## Visualizamos la curva de importancia y guardamos la imagen
png(filename=file.path(path, "Importance_Variables.png"))
plot(important_variables[1:1000]$Importance, type="s", xlab="Nº Variables", ylab="Importancia", col="red")
dev.off()

## Eliminamos las variables que no sean importantes
important_variables <- important_variables[1:num_imp_var]
data <- data[ , c(fixed_var, important_variables$Variables), with=F]

if (sum(is.na(data))==0){
  ## Guardamos los datos finales
  write.table(data, file.path(path, "final_dat.csv"), row.names=F, sep=";")
  print("Final Data Saved")
  
  ## Guardamos las variables mÃ¡s importantes
  write.table(important_variables, file.path(path, "important_variables.csv"), row.names=F, sep=";")
  print("Important Variables Saved")
}

## Traza de ejecución
final_time <- round((proc.time() - time)/60,2)
print(sprintf("Tiempo de ejecución: %s min.", final_time[3]))

