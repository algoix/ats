#Code 1
require(quantmod)
require(PerformanceAnalytics)
getSymbols("SPY")
chartSeries(SPY, TA=NULL)
data=SPY[,4]
macd = MACD(data, nFast=12, nSlow=26,nSig=9,maType=SMA,percent = FALSE)
chartSeries(data, TA="addMACD()")
signal<-Lag(ifelse(macd$macd<-macd$signal, -1, 1))
returns<- ROC(data)*signal
returns<- returns['2008-06-02/2016-10-7']
portfolio<- exp(cumsum(returns))
plot(portfolio)
table.Drawdowns(returns, top=10)
table.DownsideRisk(returns)
charts.PerformanceSummary(returns)

#CODE 2 

install.packages("quantstrat", repos="http://R-Forge.R-project.org")
install.packages("blotter", repos="http://R-Forge.R-project.org")
install.packages("FinancialInstrument", repos="http://R-Forge.R-project.org")
install.packages("quantstrat", repos="http://R-Forge.R-project.org")
install.packages("blotter", repos="http://R-Forge.R-project.org")
install.packages("FinancialInstrument", repos="http://R-Forge.R-project.org")
require(quantstrat)

NSEI&lt;-ym_xts
colnames(NSEI)&lt;-c("Open","High","Low","Close")
stock.str='NSEI' # stock we trying it on
currency('INR')
stock(stock.str,currency='INR',multiplier=1)
initEq=1000
initDate = index(NSEI[1])#should always be before/start of data
portfolio.st='MeanRev'
account.st='MeanRev'
initPortf(portfolio.st,symbols=stock.str, initDate=initDate)
initAcct(account.st,portfolios='MeanRev', initDate=initDate)
initOrders(portfolio=portfolio.st,initDate=initDate)
addPosLimit(portfolio.st, stock.str, initDate, 1, 1 ) #set max pos
stratMR &lt;- strategy("MeanRev", store = TRUE)
THTFunc&lt;-function(CompTh=NSEI,Thresh=6, Thresh2=3){
  numRow&lt;- nrow(CompTh)
  xa&lt;-coredata(CompTh)[,4]
  xb&lt;-xa
  tht&lt;-xa[1]
  for(i in 2:numRow){
    if(xa[i]&gt;(tht+Thresh)){ tht&lt;-xa[i]}
    if(xa[i]&lt;(tht-Thresh)){ tht&lt;-xa[i]}
    xb[i]&lt;-tht
  }
  up &lt;- xb + Thresh2
  dn&lt;- xb-Thresh2
  res &lt;- cbind(xb, dn,up)
  colnames(res) &lt;- c("THT", "DOWN", "UP")
  reclass(res,CompTh)
}
stratMR &lt;- add.indicator(strategy = stratMR, name = "THTFunc", arguments = list(CompTh=quote(mktdata), Thresh=0.5, Thresh2=0.3), label='THTT')
stratMR &lt;- add.signal(stratMR,name="sigCrossover",arguments = list(columns=c("Close","UP"),relationship="gt"),label="Cl.gt.UpperBand")
stratMR &lt;- add.signal(stratMR,name="sigCrossover",arguments = list(columns=c("Close","DOWN"),relationship="lt"),label="Cl.lt.LowerBand")
stratMR &lt;- add.rule(stratMR,name='ruleSignal', arguments = list(sigcol="Cl.gt.UpperBand",sigval=TRUE, prefer = 'close', orderqty=-1, ordertype='market', orderside=NULL, threshold=NULL,osFUN=osMaxPos),type='enter')
stratMR &lt;- add.rule(stratMR,name='ruleSignal', arguments = list(sigcol="Cl.lt.LowerBand",sigval=TRUE, prefer = 'close', orderqty= 1, ordertype='market', orderside=NULL, threshold=NULL,osFUN=osMaxPos),type='enter')
start_t&lt;-Sys.time()
out&lt;-try(applyStrategy(strategy=stratMR , portfolios='MeanRev') )
getOrderBook('MeanRev')
end_t&lt;-Sys.time()
updatePortf('MeanRev', stock.str)
chart.Posn(Portfolio='MeanRev',Symbol=stock.str)
tradeStats('MeanRev', stock.str)
View(t(tradeStats('MeanRev')))
.Th2 = c(.3,.4)
.Th1 = c(.5,.6)
require(foreach)
require(doParallel)
registerDoParallel(cores=2)
stratMR&lt;-add.distribution(stratMR,paramset.label='THTFunc',component.type = 'indicator',component.label = 'THTT', variable = list(Thresh = .Th1),label = 'THTT1')
stratMR&lt;-add.distribution(stratMR,paramset.label='THTFunc',component.type='indicator',component.label = 'THTT', variable = list(Thresh2 = .Th2),label = 'THTT2')
results &lt;- apply.paramset(stratMR, paramset.label='THTFunc', portfolio.st=portfolio.st, account.st=account.st, nsamples=4, verbose=TRUE)
stats &lt;- results$tradeStats
View(t(stats))

#Code 3

require(depmixS4)
require(quantmod)
require(PerformanceAnalytics)
getSymbols('SPY', from = '1990-01-01', src='yahoo', adjust = TRUE)
spyRets <- na.omit(Return.calculate(Ad(SPY)))
set.seed(123)

hmm <- depmix(SPY.Adjusted ~ 1, family = gaussian(), nstates = 3, data=spyRets)
hmmfit <- fit(hmm, verbose = FALSE)
post_probs <- posterior(hmmfit)
post_probs <- xts(post_probs, order.by=index(spyRets))
plot(post_probs$state)
summaryMat <- data.frame(summary(hmmfit))
colnames(summaryMat) <- c("Intercept", "SD")
bullState <- which(summaryMat$Intercept > 0)
bearState <- which(summaryMat$Intercept < 0)
 
hmmRets <- spyRets * lag(post_probs$state == bullState) - spyRets * lag(post_probs$state == bearState)
charts.PerformanceSummary(hmmRets)
table.AnnualizedReturns(hmmRets)

##IQfeed csv data loading 
df_stock <- read.csv("~/Desktop/PROJECT_1/IQ_data/SPY.csv", header=FALSE)
SPY<-df_stock
lagOP<-lags(SPY$V3,1)[,1]
SPYCL<-lagOP[2:length(SPY$V5)]
#spyRets <-log(lagOP/SPYCL)
#spyRets <- na.omit(Return.calculate(SPY$V5))
SPYhmm<-SPY[-1,]
SPYhmm["spyRets"]<-log(lagOP/SPYCL)

hmm <- depmix(spyRets ~ 1, family = gaussian(), nstates = 3, data=SPYhmm)



