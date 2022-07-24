
###############################################################################################
###################################### AUX FUNCTIONS NBA ###################################### 
###############################################################################################

####################### One Hot Encoding ####################### 

onehot <- function(x){
  values<-unique(x);
  
  ret<-data.table(matrix(0,nrow = length(x),ncol=length(values)))
  
  ret<-data.table(foreach (i = 1:length(values),.combine=cbind)%dopar%{
    as.numeric(x==values[i])
  })
  
  colnames(ret)<-values;
  return(ret);
}

one_hot_encoding <- function(dat,categorical,no_categorical,fixed_var){
  onehot_list<-apply(dat[,categorical,with=F],2,onehot);
  temp<-foreach(i = 1:length(names(onehot_list)),.combine=cbind)%dopar%{
    var_name<-names(onehot_list)[i]
    info<-onehot_list[[var_name]];
    colnames(info)<-paste0(var_name,colnames(info));
    info;
  }
  
  if (!is.na(fixed_var)[1]){
    dat <- data.table(dat[,fixed_var,with=F],temp, dat[,no_categorical,with=F]);
  } else {
    dat <- data.table(temp, dat[,no_categorical,with=F]);
  }
  return(dat);
}


####################### Tipify ####################### 

tipify <- function(dat,m,s, remove_constant = TRUE){
  
  # Convert to data.table
  if (class(dat)[1]!="data.table"){
    dat<-data.table(dat);
  }
  
  # Remove constant variables
  constant_var<-which(s==0);
  if (length(constant_var)>0){
    if (remove_constant){
      dat<-dat[,-constant_var,with=F];
      m<-m[-constant_var];
      s<-s[-constant_var];
      print(sprintf("%s constant variables removed",length(constant_var)));
    }
    else {
      print(sprintf("%s constant variables NOT removed",length(constant_var)));
      s[constant_var] <- 1;
    }
  }
  
  dat <- as.matrix(dat);
  for (i in 1:ncol(dat)){
    dat[,i] <-  (dat[,i]-m[i])/s[i];
  }
  dat <- data.table(dat)
  
  colnames(dat)<-names(m);
  
  return(list(dat=dat,m=m,s=s));
}


####################### Important Variables ####################### 
select_important<-function(dat, target){
  varimp <- filterVarImp(x=dat[,-fixed_var, with=F], y=target, nonpara=TRUE)
  varimp <- data.table(rownames(varimp), varimp)
  colnames(varimp) <- c("Variables", "Importance")
  varimp <- varimp[order(-Importance)]
  return(varimp)
}


####################### Redundant Variables ####################### 
remove_redundant <- function(correlations, redundant_threshold){
  redundancy <- apply(correlations, 2, function(x){which(x>redundant_threshold)})
  redundancy <- redundancy[which(sapply(redundancy, length)>1)]
  
  redundant_variables<-c();
  for (i in redundancy){
    imp <- sort(correlations[1, i], decreasing = TRUE)
    redundant_variables <- c(redundant_variables, names(imp)[2:length(i)])
  }
  redundant_variables <- unique(redundant_variables)
} 
