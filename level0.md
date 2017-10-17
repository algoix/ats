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
    
    
