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

