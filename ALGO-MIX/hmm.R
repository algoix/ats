require(depmixS4)
require(quantmod)
require(PerformanceAnalytics)
library(TTR)
library(Eplot)
library(ggplot2)
library(reshape2)

getSymbols("SPY", from = '1990-01-01', src='yahoo', adjust = TRUE)
spyRets <- na.omit(Return.calculate(Ad(SPY)))
set.seed(123)

hmm <- depmix(SPY.Adjusted ~ 1, family = gaussian(), nstates = 3, data=spyRets)
hmmfit <- fit(hmm, verbose = FALSE)
post_probs <- posterior(hmmfit)
post_probs <- xts(post_probs, order.by=index(spyRets))

# Initial probabilities
summary(hmmfit, which = "prior")
# Transition probabilities
summary(hmmfit, which = "transition")
# Reponse/emission function
summary(hmmfit, which = "response")


plot(post_probs$state)
summaryMat <- data.frame(summary(hmmfit))
colnames(summaryMat) <- c("Intercept", "SD")
bullState <- which(summaryMat$Intercept > 0)
bearState <- which(summaryMat$Intercept < 0)
hmmRets <- spyRets * lag(post_probs$state == bullState) - spyRets * lag(post_probs$state == bearState)
charts.PerformanceSummary(hmmRets)
table.AnnualizedReturns(hmmRets)

# Classification (inference task)
tsp500 <- as.ts(SPY)
pbear <- as.ts(post_probs[,2])
#??tsp(pbear) <- tsp(tsp500)
plot(cbind(tsp500[, 6], pbear),main = "Posterior Probability of State=1 (Volatile, Bear Market)")

map.bear <- as.ts(posterior(hmmfit)[, 1] == 1)
#tsp(map.bear) <- tsp(tsp500)
plot(cbind(tsp500[, 6], map.bear),
     main = "Maximum A Posteriori (MAP) State Sequence")


rowSums(head(post_probs)[,2:5]) 
##
LogReturns <- log(SPY$SPY.Close) - log(SPY$SPY.Open)
ATR= ATR(SPY[,2:4], n = 15, maType="WMA")[,1]; 
ModelData<-data.frame(LogReturns,ATR) #create the data frame for our HMM model
ModelData<-ModelData[-c(1:15),] #remove the data where the indicators are being calculated
colnames(ModelData)<-c("LogReturns","ATR") #name our columns
set.seed(1)
HMM<-depmix(list(LogReturns~1,ATR~1),data=ModelData,nstates=3,family=list(gaussian(),gaussian())) #We’re setting the LogReturns and ATR as our response variables, using the data frame we just built, want to set 3 different regimes, and setting the response distributions to be gaussian.
HMMfit<-fit(HMM, verbose = FALSE) #fit our model to the data set
print(HMMfit) #we can compare the log Likelihood as well as the AIC and BIC values to help choose our model
summary(HMMfit)
HMMpost<-posterior(HMMfit) #find the posterior odds for each state over our data set
head(HMMpost) #we can see that we now have the probability for each state for everyday as well as the highest probability class.



#PLOT ATR,States,Returns...
##
# http://www.meetup.com/R-User-Group-SG/files/

## Bull and Bear Markets ##
# Load S&P 500 returns from Yahoo
Sys.setenv(tz = "EST")
sp500 <- getYahooData("^GSPC", start = 19500101, end = 20120909, freq = "daily")
head(sp500)
tail(sp500)

# Preprocessing
# Compute log Returns
ep <- endpoints(sp500, on = "months", k = 1)
sp500LR <- sp500[ep[2:(length(ep)-1)]]
sp500LR$logret <- log(sp500LR$Close) - lag(log(sp500LR$Close))
sp500LR <- na.exclude(sp500LR)
head(sp500LR)

# Build a data frame for ggplot
sp500LRdf <- data.frame(sp500LR)
sp500LRdf$Date <-as.Date(row.names(sp500LRdf),"%Y-%m-%d")

# Plot the S&P 500 returns
ggplot( sp500LRdf, aes(Date) ) + 
  geom_line( aes( y = logret ) ) +
  labs( title = "S&P 500 log Returns")


# Construct and fit a regime switching model
mod <- depmix(logret ~ 1, family = gaussian(), nstates = 4, data = sp500LR)
set.seed(1)
fm2 <- fit(mod, verbose = FALSE)
#
summary(fm2)
print(fm2)

# Classification (inference task)
probs <- posterior(fm2)             # Compute probability of being in each state
head(probs)
rowSums(head(probs)[,2:5])          # Check that probabilities sum to 1

pBear <- probs[,2]                  # Pick out the "Bear" or low volatility state
sp500LRdf$pBear <- pBear            # Put pBear in the data frame for plotting

# Pick out an interesting subset of the data or plotting and
# reshape the data in a form convenient for ggplot
df <- melt(sp500LRdf[400:500,6:8],id="Date",measure=c("logret","pBear"))
#head(df)

# Plot the log return time series along withe the time series of probabilities
qplot(Date,value,data=df,geom="line",
      main = "SP 500 Log returns and 'Bear' state probabilities",
      ylab = "") + 
  facet_grid(variable ~ ., scales="free_y")
###
##

snp <- getYahooData("^GSPC",start=19990101,end=20161025,freq="daily")
vix <- getYahooData("^VIX",start=19990101,end=20161025,freq="daily")
time <- as.Date(substr(index(snp),1,10))[-1] # get the time index and remove first obs
vixp <- apply(vix[,-5],1,mean)[-1] # average daily price (sort of..) # remove first obs

# convert to returns (that is why we removed first obs before
snpp <- as.numeric(apply(snp[,-5],1,mean)) # average daily price (sort of..)
TT <- length(snpp)
ret <-100*( snpp[2:TT]/snpp[1:(TT-1)] - 1) # convert to daily returns
par(mfrow=c(1,1))
plott(ret,vixp,ret=F,ty="p",xlab="VIX",main="Market returns (%) on VIX")
mod <- depmix(ret ~ 1, data=data.frame(ret=ret,vixp=vixp), transition=~1,nstates=2, family=gaussian())
mod2 <- depmix(ret ~ 1, data=data.frame(ret=ret,vixp=vixp), transition=~vixp,nstates=2, family=gaussian())
# mod2 <- depmix(ret ~ vixp, data=data.frame(ret=ret,vixp=vixp), nstates=2)
modfit <- fit(mod,verbose=T)
modfit2 <- fit(mod2,verbose=T)
summary(modfit)
summary(modfit2)

summary(modfit, which = "transition")
summary(modfit2, which = "transition")
slotNames(modfit2)

trans1 <- as.matrix(posterior(modfit))
trans2 <- as.matrix(posterior(modfit2))
## The Figure
par(mfrow=c(2,1))
col1 <- rgb(1,0,0,1/4)
col2 <- rgb(0,1,0,1/4)
par(mfrow=c(2,1))
plott(scale(snpp[-1],scale=T)/3,time,ret=F,ty="l",main="(scaled) SNP with superimposed posterior probabilities of bear regime")
plott(trans1[,2],time,ty="l",ret=F,col=col1,add=T)
plott(trans2[,3],time,ty="l",add=T,col=col2) # the other regime is base
legend('bottomright',legend=c("model without VIX","Model with VIX"),cex=1.3,
       col=c("red","green"),lty=1,bty="n",text.col=c("red","green"))
plott(vixp,time,ty="l",main="VIX")
##


#install.packages("RHmm", repos="http://R-Forge.R-project.org")
## RHmm is not available


#ANother method
require(DoMC)
dailyHMM <- function(data, nPoints) {
  subRets <- data[1:nPoints,]
  hmm <- depmix(SPY.Adjusted ~ 1, family = gaussian(), nstates = 3, data = subRets)
  hmmfit <- fit(hmm, verbose = FALSE)
  post_probs <- posterior(hmmfit)
  summaryMat <- data.frame(summary(hmmfit))
  colnames(summaryMat) <- c("Intercept", "SD")
  bullState <- which(summaryMat$Intercept > 0)
  bearState <- which(summaryMat$Intercept < 0)
  if(last(post_probs$state) %in% bullState) {
    state <- xts(1, order.by=last(index(subRets)))
  } else if (last(post_probs$state) %in% bearState) {
    state <- xts(-1, order.by=last(index(subRets)))
  } else {
    state <- xts(0, order.by=last(index(subRets)))
  }
  colnames(state) <- "State"
  return(state)
}
 
# took 3 hours in parallel
t1 <- Sys.time()
set.seed(123)
registerDoMC((detectCores() - 1))
states <- foreach(i = 500:nrow(spyRets), .combine=rbind) %dopar% {
  dailyHMM(data = spyRets, nPoints = i)
}
t2 <- Sys.time()
print(t2-t1)



#Ref:
#https://quantstrattrader.wordpress.com/tag/r/

