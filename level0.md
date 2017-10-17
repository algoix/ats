    https://gist.github.com/parthasen/e0250d4c02185640445ec9bdbf3d6b25

# AMIBROKER & IB based automatic trading
 1. This code can handle multiple stocks. 
 2. Holding increases with time or remaining cash reduced with time if matching of trades decreases.
 3. **Concept of using multiple stocks** are useful 
 4. Only pythonic datascience helps to get insight of last trading days to help AFL coding but there is no direct update from python analysis to Amibroker.

####  Real time value from IB feed.

    LP=GetRTData("Last");
    BID=GetRTData("Bid");
    BIDSIZE=GetRTData("BidSize");
    ASK=GetRTData("Ask"); 
    ASKSIZE=GetRTData("AskSize");
    H52=GetRTData("52WeekHigh");
    L52=GetRTData("52WeekLow");
    CGFRY=GetRTData("Change");
    CHP=GetRTData("High");
    CLP=GetRTData("Low");
    CLD=GetRTData("Prev");
    TV=GetRTData("TotalVolume");
    LTV=GetRTData("TradeVolume");

Above code to get real time data. We have useed the real time data to maintain minimum profit before selling. Without using this protection sometimes due to large spread between bid and ask stocks are sold in loss as our trading signal based on market price. 

**A good use to avoid slip or loss due to sudden jump:**
    
    minSProfit= (LastValue(GetRTData("Bid"))-ibc.GetPositionInfo(Name(), "Avg. cost"))*abs(IBPosSize)>5;	
  	minCProfit= (LastValue(ibc.GetPositionInfo(Name(), "Avg. cost"))-GetRTData("Ask"))*abs(IBPosSize)>5;
  	minSloss=(LastValue(stoplossS)-ibc.GetPositionInfo(Name(), "Avg. cost")*0.99)*abs(IBPosSize)>5;	
    minCloss=(LastValue(stoplossC)-ibc.GetPositionInfo(Name(), "Avg. cost")*1.01)*abs(IBPosSize)>5;


#### Market on/off
  
    MarketON=093000; TradeON = 094000; TradeOFF =153000; MarketOFF=160000; LM=155500;
    US_ON = TimeNum() >= 093000 AND TimeNum() <=160000;
    US_OFF = TimeNum() > 160000 ;//day's over*/
    US_Trade_On=TimeNum() >= 094000 AND TimeNum() <=153000;
    MarketLM=TimeNum() >=155500 AND Now( 4 )<=160000;

Above code for setting trading time on/off. Market on at 9:30 AM but trading will be on at 9:40AM. We avoid first 10 minutes of market spike/gap up or down. ( we have direction prediction to help here).
New trading will be stopped at 3:30PM ( 30 minutes before market closing. For last 30 minutes only closing trade will be on. 


#### To handle asset of different country LSE, Japan    
    ABName=IBName=Name();//getfndata("Alias");
    //IBName =Name();//getfndata("Alias");

#### Strategy
1. will filter trading and momentum stocks

2. Buy and sell short stocks
    
        filterb=B1+B2+B3+LRPup+candleAbuy;
        filters=S1+S2+S3+LRPdown+candleAsell;
        "FilterB"+filterb+"FilterS"+filters;
        FB=upline AND filterb>filters+2;//filters;
        FSS=downline AND filters>filterb+2;//filterb;
        "BUY**"+FB+"Sell**"+FSS;

3. Four parts are combined to build the strategy. 
Candle stick based indentification ( candleAbuy or candleAsell), inbuilt indicator ( RSI,MACD...) based identification, synthetic indicators and price level based elimination( to avoid gap up/down,broken up/down price level of any stock).

Eliminate stocks based on self prepared indicators. Here indicator is prepared based on estimation using time series regression and moving average functions(TSF() and MA()).

#### Candlestick

    candlesell =   IIf(KBR OR EveningDojiStar OR EveningStar OR GraveStoneDoji OR Bear3Formation OR BearishAbandonedBaby
        OR BearishBeltHold OR BearishCounterAttack OR BearishHaramiCross OR BearishHarami OR BearishSeparatingLine
        OR DarkCloudCover OR EngulfingBear OR HangingMan OR ShootingStar OR ThreeBlackCrows OR TriStarBottom 
        OR TweezerTops OR UpsideGapTwoCrows,1,0);//weight 2
    candlebuy=IIf( MorningStar OR MorningDojiStar OR Bull3Formation
        OR BullishAbandonedBaby OR BullishBeltHold OR BullishCounterAttack OR BullishHaramiCross
        OR BullishHarami OR BullishSeparatingLine OR DragonflyDoji OR EngulfingBull OR Hammer OR InvertedHammer
        OR PiercingLine OR SeperatingLines OR ThreeWhiteSoldiers OR TriStarTop OR TweezerBottoms OR KBL, 1, 0);
        
    candleAbuy=ICHIMOKUbuy+candlebuy; //Max value 6 and min 0
    candleAsell=ICHIMOKUsell+candlesell;
    "Candle Buy"+candleAbuy+"candle Sell"+candleAsell;

Candle stick based elimination of stock is based on basic theory of bearish and bullish type of candles. Each candle is of 5sec.

#### price level
Another part of strategy is to eliminate stocks those are below/above the middle of peak or based on support/resistance. If downward pattern then below else with upward price pattern and less than 1.2% of peak then above.  
    
    downline= O<(xPK1+xTr1)/2 AND xPK1<=xPK2 ;//*
    upline= xTr1>=xTr2 AND O>(xPK1+xTr1)/2 AND C<xPK1*1.2;
    "xTr1"+xTr1+"xTr2"+xTr2+"xPK1"+xPK1+"xPK2"+xPK2+"Downline"+downline+"upline"+upline;
   
#### synthetic indicator

    B1=MA(C,20)>x_est_last;
    S1= MA(C,20)<x_est_last;
    B2=(JJ-Ref(JJ,-1))/(C-Ref(C,-1))>0;
    S2=(JJ-Ref(JJ,-1))/(C-Ref(C,-1))<0;
    
    B3=Cross( MACD(2,8 ), Signal (4,10,2 ) )+Cross( StochK( 8, 2 ), StochD( 10,2,2))+ (RSI(4)>20 AND RSI(4)<75)
	+Cross(EMA(C,2),EMA( C,8 ))+Cross(MA(OBV(),2),MA(OBV(),8))+Cross(MA(PDI(),2),MA(MDI(),8))+Cross(CCI(2),CCI(8));
    S3=Cross(Signal (4,10,2 ),MACD(2,8 ))+Cross(StochD( 10,2,2 ), StochK(8,2 ))+(RSI(4)>40 AND RSI(4)<90)+Cross(EMA(C,8),EMA( C,2 ))
    +Cross(MA(OBV(),8),MA(OBV(),2))+Cross(MA(MDI(),8),MA(PDI(),2))+Cross(CCI(2),CCI(8));
    
## Selection
#### selection.afl
selection code is to eliminate stocks those have beta less than 0.6,return of change above 1% ( buy) or less than 1% ( sell). So first level of elimination of stocks is through selection.afl code before going to trade.afl. 
First level of elimination is through three parts. Two parts are beta and ROC already mentioned. Last part is based on index and ETF performance. 
In this code SPX,NDX,OCX and RUT indices with TQQQ/SQQQ ETFs are used to find the level of index

#### Beta calculation
	Beta=((20* Sum(ROC( C,60) * ROC(P_SPX,60),20)) - (Sum(ROC(C,60),20) * Sum(ROC( P_SPX,60),20))) / ((20* Sum((ROC(P_SPX,60)^2 ),20)) - (Sum(ROC(P_SPX,60),20)^2 ));
	cons=(MA(Beta,50)>0.6 AND Correlation(P_SPX,C,50)>0.6);
#### ROC calculation
ROC(C,12)
#### ETF performance
	downline=retNDX60<0 AND  retSQQQ20>0 AND retSQQQ20>retTQQQ20 AND Trough( P_SPX,0.2, 1 )<Trough( PL_SPX,0.2, 2) AND MA(P_SPX,20)>LastValue(Trough( P_SPX, 2, 1 )) AND Peak( P_SPX,0.2, 1 )<Peak( P_SPX,0.2, 2 );//*
	upline=retNDX60>0 AND retTQQQ20>0 AND retSQQQ20<retTQQQ20 AND MA(P_SPX,20)>LastValue(Peak( P_SPX,0.2, 1 )*0.99) and MA(P_SPX,20)<LastValue(Peak( P_SPX,0.2, 1 )*1.05) AND Peak( P_SPX,0.2, 1 )> Peak( P_SPX,0.2, 2 ) AND Trough( P_SPX, 0.2, 1 )>Trough( P_SPX, 0.2, 2);
	//AND retVIX20>0  AND retOEX20<0.2 AND retSPX20<0.2 AND retRUT20<0.2 AND
	//AND retVIX120<0  retOEX20>0.3 AND retSPX20>0.2 AND retRUT20>0.2 AND  

#### Hypothesis
We will consider uptrend market in the range of below 1% from peak to 5% above peak along with TQQQ change should be higher than SQQQ and NDX change should be above 0 to consider BUY type market.
We will consider downtrend market, price above trough, NDX change below 0, SQQQ above 0 and SQQQ change above TQQQ change. 
So trend,support and resistance concepts are used here.

### Elimination/Selection using LIST
**InWatchList()** function is used in AFL code to list out based on the output of different codes. Those lists are to eliminate/separate stocks to trade at last.
**New stocks**
LIST 4 is used to fill manually to store buy catagory stocks. Whereas LIST 6 is used to store sell catagory stocks. Both are to be filled manually based on our stock selection method.
Selection.afl code is first to eliminate stocks based on beta,ROC,stock and index level.

	LIST 4 -> LIST 12	(BUY)
	LIST 6 -> LIST 9 	(SELL)
	LIST 12+ Buy strategy -> buy signal to TWS
	LIST 9 + Sell strategy -> sell short signal to TWS
**Holding Stocks/Portfolio**
LIST 17 is to store the stocks in portfolio. These stocks are further catagorized based on volume. If volume of any holding stock to 25 min average of volume is in the range of 2 to 1.5 then that holding stock will be in LIST 31.

	LIST 17-> LIST 30 (>2)
	LIST 17-> LIST 31 ( 2 – 1.5)
	LIST 17-> LIST 32 ( 1.5 – 0.8)
	LIST 17-> LIST 33 ( <0.8)
	
This catagorisation helped us to set different target profit for selling the holding stocks. Here the sentiment based differention will be used later.	
**ETFs** 
SQQQ and TQQQ ETFs will be stored at LIST 64.

Weighted Stocks
Stocks that are special with the expectation of higher return. Like, those stocks have better expected earning then we can keep those in these lists. 

	LIST 18 = Weighted BUY
	LIST 19 = Weighted SELL
	
	

	

