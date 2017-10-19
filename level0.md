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
	
##### Trading Signal
	Buysig=(InWatchList(12)  AND FB) OR (InWatchList(18) AND O>EMA(C,12)) OR  (InWatchList(64) AND ROC(C,120)<-2 AND ROC(C,20)<ROC(C,60) AND ROC(C,4)<0) ;//AND O>valB AND O<valT;// second data 
	Sellsig=(inwatchlist(9) AND FSS)  OR (InWatchList(19) AND C<EMA(O,12));// AND O>valT ;

1. FB and FSS are strategies built on self prepared indicators, built in indicators,candle sticks and price support/resistance.
2. Strategy and LISTs are used to send signals. Strategies for stocks,ETFs and weighted stocks are different. Those signals are used for BUY/Short Sell.  But COVER and SELL signals are based on the target percentage above average holding price. 
Target percentage is based on volume profile ( <0.8 to >2 of volume to average volume of last 25 minutes).
3. Target -> minimum profit target $5 for each holding. If volume is above or if we consider highly liquid then that minimum profit target increases to $12.5.
This method is applicable to cover (minCProfit) and sell(minSProfit). 

##### TWS account details
	ibc = GetTradingInterface("IB");
	IBcStatus = ibc.IsConnected();
	IBPosSize = ibc.GetPositionSize( IBName );
	IBcStatusString = WriteIf(IBCStatus==0,"TWS Not Found",WriteIf(IBCStatus==1,"Connecting to TWS",WriteIf(IBCStatus==2,"TWS OK",WriteIf(IBCStatus==3,"TWS OK (msgs)",""))));
	printf("Net Liquidity or balance:");
	liqCAD=ibc.GetAccountValue("[CAD]NetLiquidation");
	printf("Net Liquidation USD :");
	CashBalanceStr = ibc.GetAccountValue("[USD]NetLiquidationByCurrency");
	printf("Net Liquidation CAD:");
	CashBalanceCAD = ibc.GetAccountValue("[CAD]NetLiquidationByCurrency");
	printf("Available Funds USD:"); 
	USDaf=ibc.GetAccountValue("[USD]AvailableFunds");
	printf("Available Fund CAD:"); 
	CADaf=ibc.GetAccountValue("[CAD]AvailableFunds");
	printf("Total Cash Balance USD:"); 
	USDtcb=ibc.GetAccountValue("[USD]TotalCashBalance");
	printf("Total Cash Balance CAD:"); 
	CADtcb=ibc.GetAccountValue("[CAD]TotalCashBalance");
	printf("AvailableFunds CAD:");
	availablefundstr=ibc.GetAccountValue("[CAD]AvailableFunds");
	printf("Excess Liquidity CAD:");
	excessfundstr=ibc.GetAccountValue("[CAD]ExcessLiquidity");
	printf("MaintenanceMarginReqCAD:");
	marginfundUSD=ibc.GetAccountValue("[CAD]MaintMarginReq");
	printf("Equity With Loan Value CAD:");
	loanfundUSD=ibc.GetAccountValue("[CAD]EquityWithLoanValue");
	printf("Buying Power CAD or  excessfund:");
	excessfundCAD=ibc.GetAccountValue("[CAD]BuyingPower");

	if (liqCAD == "")
    Balance = 0;
	else
    Balance = StrToNum(liqCAD);
    
	if (excessfundCAD =="")
    excessfund = 0;
	else
   	excessfund = StrToNum(excessfundCAD);  
   
 	AccountCutout =excessfund<10000; 
CAD and USD liquidity, cash balance, total cash balance ... are visible and can be tracked from Amibroker window. This is helpful for tracking. Although excessfundCAD is used to automatic cutoff from trading when banlance of excessfund is low (below $10000 or 10% of excessfund in CAD).

##### ORDER SIZE
	function stockprice()
	{base=1;
	for(i=0;i<=LastValue(C);i++){
	base++;
	}
	roundoff=int(base/5)*5;
	return lastvalue(roundoff);
	}

	base=stockprice();
	printf("\n"+"Roundoff Price"+base);
	IBOrderSize =(int((Balance*0.01 + excessfund*0.01)/base)/100)*100;//(int((Balance*0.02 + excessfund*0.03)/base)/100)*100;
	printf("\n"+"unfiltered Order size"+IBOrderSize);
	ordersize=IIf(IBOrderSize<(Balance*0.085/base) AND IBOrderSize>(Balance*0.02/base) AND AccountCutout==0 ,IBOrderSize,0);
	printf("\n"+"Maximum Order size"+Balance*0.085/base);
	printf("\n"+"Minimum Order size"+Balance*0.02/base);
	printf("\n"+"Actual:"+ordersize+"\n");
	//openpos = ibc.GetPositionList(); 
	pendinglist=ibc.GetPendingList( 0, "Pending" );  
	averageprice=0;

Stock price is first rounded off to integer. That rounded off price is used at denominator where numerator is summation of 1% balance and 1% excessfund. This oredersize is again wrapped to control low size by minimum size above 2% of balance and maximum size below 8.5% of balance.
So order size will be reducing with reducing balance but it will be in the range of 2% of balance to 8.5% of balance.	

##### STOP LOSS
Stop loss is based on ATR range.

	stoplossS=C-Max(ATR(5)*15,MA(C,200)*0.004);
	stoplossC=C+Max(ATR(5)*15,MA(C,200)*0.004);

	minSloss=(LastValue(stoplossS)-ibc.GetPositionInfo(Name(), "Avg. cost")*0.99)*abs(IBPosSize)>5;	
  	minCloss=(LastValue(stoplossC)-ibc.GetPositionInfo(Name(), "Avg. cost")*1.01)*abs(IBPosSize)>5;

Stop loss is effective only when the loss amount is above $15 for each holding either long/short. RISK control
Time for new trading will be on from 9:40 AM to 3:30 PM but closing trading will be on for whole market time.
Trading off due to account cutout. Account cut out is set at $10000 CAD. Later plan to off the trading at 35% of net liquidity.If fund value is $x then account cutout will be set at $0.35x.
Controlled order size based on existing balance and excess fund. Size will be in the range of 2% of balance to 8.5% of balance.  
Stop loss is based on ATR range to make the target dynamic, at the same time minium value is also fixed ($15) to avoid unexpected trade at narrow range of ATR.

##### SIGNAL from Amibroker to TWS

	if(B1 AND AccountCutout==0)  //BUY
    {
        OID= ibc.PlaceOrder( Name(), "BUY",Size, "MKT",0, 0, "Day", True); 
        ORderStatus = ibc.GetStatus( OID, True);
        if(ORderStatus == "Filled"){
        StaticVarSetText("OrderID"+ABName,OID);
        }
        for (dummy=0; dummy<40; dummy++) ibc.Sleep(50);  //Usually takes up to about a second for TWS to get acknowledgement from IB
        if (SubmitOrders)
        {
            for (dummy=0; dummy<40; dummy++) ibc.Sleep(50);  //Usually takes up to about a second for TWS to get acknowledgement from IB

             tradetime=GetPerformanceCounter()/1000; 
             while ((GetPerformanceCounter()/1000 - tradetime) <5) // give up after 5 seconds
             {
                 ibc.Reconnect();  //Refreshes ibc, and gets accurate status
                 //ORderStatus = ibc.GetStatus( OID, True);
                 if (ORderStatus == "PreSubmitted" || ORderStatus == "Submitted" || ORderStatus == "Filled")
                     break;
             }
        }
                     
    }

This code to transmit signal of buy/sell/short sell/cover generating from Amibroker to Interactive Broker Terminal.

