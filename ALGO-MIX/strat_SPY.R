## NEED TO RUN DS_SPY_.R for feature generation of ZN and J

## Trend ZN and Jump J are needed for [-1,+1] for momentum
## ZN,J [0] for MR market
## a caution line for CL_B and CL_SH beyond which trade will be stopped 


#Mean Reversion : spread_MR>0.022 this increases SH value
MR <- ifelse(roc3>UL_roc+0.00005|spread_MR>45,"SH",ifelse(spread_MR<11|roc3<LL_roc,"B",0))
MR <- ifelse(MR=="SH" & MS=="T","SH",ifelse(MR=="B" & MS=="T","B",0))
spread <- VWAP-price# mean reversion
#Momentum [18,68,15]
MM <- ifelse(TR>SIC_H_SPY+0.005|TR>R2+0.005,"B",ifelse(TR<SIC_L_SPY|TR<S2,"SH",0))
MM <-ifelse(MM=="SH" & MS=="MD","SH",ifelse(MM=="B" & MS=="MU","B",0))
trade <- ifelse(MR=="SH"|MM=="SH","SH",ifelse(MR=="B"|MM=="B","B",0))


## warning== hedging/no trade/manual strade of SPY 
# State change of price SPY by +- 1 and reflection of SPXL,SPXS for hedging.Hedging when holding crosses 3*order size. If holding of SPY more than -300 and price >0.5 of holding then SPXL.
## spread +- 0.01 and SC ==[7,86,7]
spread_SC <- rollapply(price-(SMA(TR,1000)+1), width =12,FUN = sum,fill=NA, align = "right")
#SC <- ifelse(spread_SC>25 & J==1,"SCU",ifelse(spread_SC<5 & J==-1,"SCD",0))
