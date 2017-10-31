### AFL
##### setup

    _SECTION_BEGIN("BACKTESTING");
    SetOption("ExtraColumnsLocation",1);
    OptimizerSetEngine("cmae");
    SetBacktestMode(backtestRegular);
    SetBacktestMode(backtestRegularRawMulti);
    SetOption("initialequity",15000);
    MaxPos=1;
    SetOption("maxopenpositions",MaxPos);
    SetPositionSize(2000,spsValue);
    SetOption("CommissionMode",2);//$
    SetOption("CommissionAmount",0);
    SetTradeDelays(0,0,0,0);
    BuyPrice=SellPrice=ShortPrice=CoverPrice=Close;
    SetChartOptions(0,chartShowArrows|chartShowDates);
    //_N(Title = StrFormat("{{NAME}} - {{INTERVAL}} {{DATE}} Open %g, Hi %g, Lo %g, Close %g (%.1f%%) {{VALUES}}", O, H, L, C ));
    _SECTION_END();
    
