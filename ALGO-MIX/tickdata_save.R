#saving data in .dat format is effective. Pyhton analysis based on this file already done
#see IBPy_marketdata_collection.ipynb
library(IBrokers)
tws <- twsConnect()
##
# write to an open file connection
fh <- file('/home/octo/Desktop/PROJECT_1/data_R/SPY.dat',open='a')
reqMktData(tws,twsEquity("SPY"), file=fh)
close(fh)
##




#Saving in csv format 
##
spy.csv <- file("/home/octo/Desktop/PROJECT_1/data_R/SPY.csv", open="w")

# run an infinite-loop ( <C-c> to break )
reqMktData(tws, twsSTK("SPY"), 
           eventWrapper=eWrapper.MktData.CSV(1), 
           file=spy.csv)
#reqMktData(tws,Contract = twsSTK("SPY"),eventWrapper = eWrapper(TRUE))
# multiple stocks
#reqMktData(tws, list(twsSTK("MSFT"),twsSTK("AAPL")))
#reqMktData(tws, twsEquity("SPY"))#infinite loop
close(spy.csv)
close(tws)

##
##Ordering confirmation checked
id <- reqIds(tws)              # Important: get new id for each order.
placeOrder(tws,twsEquity("SPY"),twsOrder(id,action = "BUY",        # Or use "SELL".
    totalQuantity = "10",orderType = "MKT"))

cancelOrder(tws, id)

twsDisconnect(tws)