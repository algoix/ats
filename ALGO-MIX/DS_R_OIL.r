library(quantmod); library(TTR); library(caret);library(corrplot);library(pROC);library(FSelector);

##ROC  
dataset_ROC<-function(stock){
  dataset<-stock
  roc1<-na.omit(ROC(df_stock$V5))
  roc12<-na.omit(ROC(df_stock$V5,12))
  roc60<-na.omit(ROC(df_stock$V5,60))
  roc_sd<-na.omit(rollapply(roc12, width =12,FUN = sd,fill=NA, align = "right"))
  UL_roc=roc60+2.5*roc_sd[38:length(roc_sd)]
  LL_roc=roc60-2.5*roc_sd[38:length(roc_sd)]
  
  roc1<-roc1[60:length(roc1)]
  roc12<-roc12[49:length(roc12)]
  roc_pro <- data.frame(roc1,roc12,roc60,UL_roc,LL_roc)
  dataset = na.omit(roc_pro)
}

##pivot
#if Open price above R1 and crosses VWAP then buy and open above R2 then sell.if VWAP above R2+0.04 or above mean+2*SD+0.05 line the sell short 
dataset_RS<-function(df_stock){
  stock<-df_stock
  R1<-price*2-stock$V3
  R2<-price+(stock$V2-stock$V3)
  S1<-price*2-stock$V2
  S2<-price-(stock$V2-stock$V3)
  sd<-rollapply(stock$V3, width =12,FUN = sd,fill=NA, align = "right")
  UL<-price+1.1*sd
  LL<-price-1.1*sd
  SL<-price+1.2*sd
  CL<-price-1.2*sd
  data_RS<- data.frame(R1,R2,S1,S2,UL,LL,SL,CL)
  dataset = na.omit(data_RS)
}

##ATR
#if TR crosses up H then B or crosses down L the SH 
ATR15 <- ATR(df_stock[,c("V2","V3","V5")], n = 15, maType="WMA")[,1];
ARC_SPY <- 1.3*ATR15
HHV <-lag(na.omit(rollapply(df_stock$V2, width =15,FUN = max,fill=NA, align = "right"),1))
LLV <- lag(na.omit(rollapply(df_stock$V3, width =15,FUN = min,fill=NA, align = "right"),1))
SIC_H_SPY <- HHV-ARC_SPY[15:length(ARC_SPY)]
SIC_L_SPY <- LLV+ARC_SPY[15:length(ARC_SPY)]
SIC_up <- TR[15:length(TR)]-SIC_H_SPY
SIC_up <- ifelse(SIC_up>0,"B",0)
SIC_dn <- TR[15:length(TR)]-SIC_L_SPY
SIC_dn <- ifelse(SIC_dn<0,"SH",0)

## force Index
FI <- ifelse(forceindex>45,"B",ifelse(forceindex< 45*-1,"SH","N"))


##UWTI
#function 1
dataset<-dataset_ROC(df_stock)
DWIT_BS<-ifelse(dataset$roc1<dataset$LL_roc,"B",ifelse(dataset$roc1>dataset$UL_roc,"SH","N"))
y<-DWIT_BS
cbind(freq=table(y), percentage=prop.table(table(y))*100)
y<-tail(DWIT_BS,1000)
cbind(freq=table(y), percentage=prop.table(table(y))*100)

#function 2
dataset <- dataset_RS(df_stock)
SPY_BSH<-ifelse(dataset$VWAP>dataset$R1*1.00005,"B",ifelse(dataset$VWAP<dataset$S1*0.99995,"SH","N"))
SPY_SC<-ifelse(dataset$VWAP>dataset$R2,"S",ifelse(dataset$VWAP<dataset$S2,"C","N"))
SPY_SHB<-ifelse(dataset$VWAP>dataset$SL,"S",ifelse(dataset$VWAP<dataset$CL,"C","N"))
SPY_CS<-ifelse(dataset$VWAP>dataset$UL,"B",ifelse(dataset$VWAP<dataset$LL,"SH","N"))
#function 3

#^^ avrage distribution 23,54,23


##DWTI
df_stock <- read.csv("~/Desktop/PROJECT_1/IQ_data/UWTI.csv", header=FALSE)
#function 1
dataset<-dataset_ROC(df_stock)
UWIT_BS<-ifelse(dataset$roc1<dataset$LL_roc,"B",ifelse(dataset$roc1>dataset$UL_roc,"SH","N"))
y<-UWIT_BS
cbind(freq=table(y), percentage=prop.table(table(y))*100)
y<-tail(UWIT_BS,1000)
cbind(freq=table(y), percentage=prop.table(table(y))*100)