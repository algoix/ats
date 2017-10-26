##### Expected line
1. price 12 sec back =P_12
2. Expected move up or down. If present price below 1m line then down. Vice varsa.
3. Expected change in 5m == 0.30. in 12 sec 0.01
4. Present expected price P_12+-0.01.

        expected_price=IIf(MA(O_SPY,12)>C_1Min_SPY,C_1Min_SPY+0.01,C_1Min_SPY-0.01);

Other synthetic price lines are

        price=(Ref(H_SPY,-1)+Ref(L_SPY,-1)+Ref(C_SPY,-1))/3;
        //VWAP spread indicator with steer 
        totalVolume = Sum(Ref(v_SPY,-1),60);
        VWAP = Sum (price *Ref(v_SPY,-1),60) /totalVolume;

        upside_state_price=ValueWhen(Cross(price,VWAP),O_SPY);
        downside_state_price=ValueWhen(Cross(VWAP,price),O_SPY);
        short_state_price=IIf(price>VWAP,upside_state_price,downside_state_price);

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

