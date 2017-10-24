library(quantmod); library(TTR); library(caret);library(corrplot);library(pROC);library(FSelector);
df_stock <- read.csv("~/Desktop/PROJECT_1/IQ_data/SPY.csv", header=FALSE)

#Strategy: if Open price above R1 and crosses VWAP then buy and open above R2 then sell.if VWAP above R2+0.04 or above mean+2*SD+0.05 line the sell short 
SPY<-df_stock
price<-(SPY$V2+SPY$V3+SPY$V5)/3
R1<-price*2-SPY$V3
R2<-price+(SPY$V2-SPY$V3)
S1<-price*2-SPY$V2
S2<-price-(SPY$V2-SPY$V3)
VP<-rollapply(price*SPY$V7, width =12,FUN = mean,fill=NA, align = "right")
V<-rollapply(SPY$V7, width =12,FUN = mean,fill=NA, align = "right")
VWAP<-VP/V
sd<-rollapply(SPY$V3, width =12,FUN = sd,fill=NA, align = "right")
UL<-price+1.2*sd
LL<-price-1.2*sd
SL<-price+2*sd
CL<-price-2*sd

R11<-ifelse(VWAP>R1,"B",ifelse(VWAP<S1,"SH","N"))
R22<-ifelse(VWAP>R2,"S",ifelse(VWAP<S2,"C","N"))

SB<-ifelse(VWAP>SL,"SH",ifelse(VWAP<CL,"B","N"))
CS<-ifelse(VWAP<UL,"C",ifelse(VWAP>LL,"S","N"))

## Compute the various technical indicators that will be used 
# Force Index Indicator
forceindex <- (df_stock$V5 - df_stock$V4) * df_stock$V7 ; 
forceindex <- c(NA,head(forceindex,-1)) ;


# Buy & Sell signal Indicators (Williams R% and RSI)
RSI12  = RSI(df_stock$V5, n = 12,maType="WMA");
RSI12 = c(NA,head(RSI12,-1));
RSI40 = RSI(df_stock$V5, n = 40,maType="WMA");
RSI40 = c(NA,head(RSI40,-1));
# Price change Indicators (ROC and Momentum)
ROC12 <- ROC(df_stock$V5, n = 12,type ="discrete")*100;
ROC12 <- c(NA,head(ROC12,-1));
ROC40 <- ROC(df_stock$V5, n = 40,type ="discrete")*100;
ROC40 <- c(NA,head(ROC40,-1));
# Volatility signal Indicator (ATR)
ATR12 = ATR(df_stock[,c("V2","V3","V5")], n = 12, maType="WMA")[,1];
ATR12 = c(NA,head(ATR12,-1));
ATR40 = ATR(df_stock[,c("V2","V3","V5")], n = 40, maType="WMA")[,1];
ATR40 = c(NA,head(ATR40,-1));
#ROC upROC and dnROC

ROC30_p <- ROC(price, n = 30,type ="discrete");
ROC30_p <- c(NA,head(ROC30_p,-1));
ROC15_p <- ROC(price, n = 15,type ="discrete");
ROC15_p <- c(NA,head(ROC15_p,-1));
ROC3_p <- ROC(price, n = 3,type ="discrete");
ROC3_p <- c(NA,head(ROC3_p,-1));

sdROC <- rollapply(ROC15_p, width =15,FUN = sd,fill=NA, align = "right");
sdROC <- c(NA,head(sdROC,-1));
upROC <- ROC30_p+2*sdROC;
dnROC <- ROC30_p-2*sdROC;
#lineROC<-ifelse(ROC3_p>upROC,"B",ifelse(ROC3_p<dnROC,"SH","N"));

# ATR line
ATR15 = ATR(df_stock[,c("V2","V3","V5")], n = 15, maType="WMA")[,1];
ATR15 = c(NA,head(ATR15,-1));
ARC=

###for R11
## Combining all the Indicators and the Class into one dataframe
dataset <- data.frame(R11,forceindex,RSI12,RSI40,ROC12,ROC40,ATR12)
dataset = na.omit(dataset)

## Understanding the dataset using descriptive statistics
print(head(dataset),5)
dim(dataset)
y <- dataset$R11#CHANGE TO SB CS
cbind(freq=table(y), percentage=prop.table(table(y))*100)
z <- dataset$R22
cbind(freq=table(z), percentage=prop.table(table(z))*100)
summary(dataset)

#reserch only
y<-R1
cbind(freq=table(y), percentage=prop.table(table(y))*100)

##  Visualizing the dataset using a correlation matrix
correlations = cor(dataset[,c(2:17)])
print(head(correlations))
corrplot(correlations, method="circle")


## Selecting features using the random.forest.importance function from the FSelector package
set.seed(5)
weights <- random.forest.importance(R11~., dataset, importance.type = 1)
print(weights)

set.seed(5)
subset = cutoff.k(weights, 10)
print(subset)

## R11:RSI12,forceindex,ROC40,ATR12

## Creating a dataframe using the selected features
dataset_rf = data.frame(R11,forceindex,RSI12,ROC40,ATR12)
dataset_rf = na.omit(dataset_rf)


# Resampling method used - 4-fold cross validation 
# with "Accuracy" as the model evaluation metric.
trainControl <- trainControl(method="cv", number=4)
metric = "Accuracy"


## Trying four different Classification algorithms
# k-Nearest Neighbors (KNN)
set.seed(5)
fit.knn <- train(class~., data=dataset_rf, method="knn",metric=metric, preProc=c("range"),trControl=trainControl)

# Classification and Regression Trees (CART)
set.seed(5)
fit.cart <- train(class~., data=dataset_rf, method="rpart",metric=metric,preProc=c("range"),trControl=trainControl)

# Naive Bayes (NB)
set.seed(5)
fit.nb<-train(class~., data=dataset_rf, method="nb",metric=metric, preProc=c("range"),trControl=trainControl)

# Support Vector Machine with Radial Basis Function (SVM)
set.seed(5)
fit.svm <- train(class~., data=dataset_rf, method="svmRadial",metric=metric,preProc=c("range"),trControl=trainControl)

## Evaluating the algorithms using the "Accuracy" metric
results <- resamples(list(KNN=fit.knn,CART=fit.cart, NB=fit.nb, SVM=fit.svm))
summary(results)
dotplot(results)

## Tuning the shortlisted algorithm (KNN algorithm)
set.seed(5)
grid <- expand.grid(.k=seq(1,10,by=1))
fit.knn <- train(class~., data=dataset_rf, method="knn", metric=metric, tuneGrid=grid,preProc=c("range"), trControl=trainControl)
print(fit.knn)


