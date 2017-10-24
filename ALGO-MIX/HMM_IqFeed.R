library(quantmod); library(TTR); library(caret);library(corrplot);library(pROC);library(FSelector);
require(depmixS4)
require(PerformanceAnalytics)
library(Eplot)
library(ggplot2)
library(reshape2)

#HMM
ModelData<-data.frame(spread_SPY,spread_SC) #create the data frame for our HMM model
ModelData<-tail(ModelData,5000) #remove the data where the indicators are being calculated
colnames(ModelData)<-c("Spread","ATR") #name our columns
set.seed(1)
HMM<-depmix(list(Spread~1,ATR~1),data=ModelData,nstates=3,family=list(gaussian(),gaussian())) #We???re setting the LogReturns and ATR as our response variables, using the data frame we just built, want to set 3 different regimes, and setting the response distributions to be gaussian.
##HMM<- depmix(Spread~1, family = gaussian(), nstates = 3, data=ModelData)
HMMfit<-fit(HMM, verbose = FALSE) #fit our model to the data set
print(HMMfit) #we can compare the log Likelihood as well as the AIC and BIC values to help choose our model
summary(HMMfit)
HMMpost<-posterior(HMMfit) #find the posterior odds for each state over our data set
head(HMMpost) #we can see that we now have the probability for each state for everyday as well as the highest probability class.
HMMpost<- xts(HMMpost, order.by=tail(df_SPY$V1,5000))
# Initial probabilities
summary(HMMpost, which = "prior")
# Transition probabilities
summary(HMMpost, which = "transition")
# Reponse/emission function
summary(HMMpost, which = "response")
plot(HMMpost$state)
summaryMat <- data.frame(summary(HMMfit))
colnames(summaryMat) <- c("Intercept", "SD")
bullState <- which(summaryMat$Intercept > 0)
bearState <- which(summaryMat$Intercept < 0)
hmmRets <- tail(df_SPY[,c("V1","V4")],5000) * lag(HMMpost$state == bullState) - tail(df_SPY[,c("V1","V4")],5000)* lag(HMMpost$state == bearState)
# need time series convertion before this
charts.PerformanceSummary(hmmRets)
table.AnnualizedReturns(hmmRets)
# Classification (inference task)
tsp500 <- as.ts(df_SPY)
pbear <- as.ts(HMMpost[,2])
#??tsp(pbear) <- tsp(tsp500)
plot(cbind(tsp500[,4], pbear),main = "Posterior Probability of State=1 (Volatile, Bear Market)")
