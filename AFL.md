##### Expected price
1. short market state `price>VWAP`
2.  expected_change = last day's high-low difference per minute plus todays price change from starting price at 50th bar per minute.
3. short market state above or blow 5 min line `C_5Min_SPY_tsf` then add/substract

        price=(Ref(H_SPY,-1)+Ref(L_SPY,-1)+Ref(C_SPY,-1))/3;
        //VWAP spread indicator with steer 
        totalVolume = Sum(Ref(v_SPY,-1),60);
        VWAP = Sum (price *Ref(v_SPY,-1),60) /totalVolume;
        upside_state_price=ValueWhen(Cross(price,VWAP),O_SPY);
        downside_state_price=ValueWhen(Cross(VWAP,price),O_SPY);
        short_state_price=IIf(price>VWAP,upside_state_price,downside_state_price);
        //expected change
        starting_price=ValueWhen(Bars_so_far_today==50,O_SPY); 
        time_lapsed=NumToStr(timenum()/100);
        hh_t = StrToNum(StrLeft(time_lapsed,3))-10;
        mm_t= StrToNum(StrMid(time_lapsed,3,2))+30;
        time_lapsed=round((hh_t*60+mm_t));
        expected_change=(C_1Min_SPY-starting_price)/time_lapsed +((H_YDay_SPY-H_YDay_SPY)/2)/400;
        C_5Min_SPY_tsf=TSF(C_5Min_SPY,12);
        expected_price=IIf(short_state_price>C_5Min_SPY_tsf,C_1Min_SPY+expected_change,C_1Min_SPY-expected_change);

##### Expected market line
If market below `expected_market_price` then market down.

        starting_price=ValueWhen(Bars_so_far_today==500,O_SPY); 
        expected_change=(H_YDay_SPY-L_YDay_SPY)/400;
        printf("starting_price:"+"\t"+starting_price+"\n");
        printf("last closing price:"+"\t"+C_YDay_SPY+"\n");
        time_lapsed=NumToStr(timenum()/100);
        hh_t = StrToNum(StrLeft(time_lapsed,3))-10;
        mm_t= StrToNum(StrMid(time_lapsed,3,2))+30;
        time_lapsed=round((hh_t*60+mm_t));
        printf("time_lapsed:"+"\t"+time_lapsed+"\n");
        for(i=0;i<time_lapsed;i++)
        {
        expected_market_price=starting_price;
        expected_change_np=IIf(starting_price>C_YDay_SPY,expected_change,-1*expected_change);
        expected_market_price=expected_market_price+i*expected_change_np;
        }




##### sentiment_upline and sentiment_dnline

