##### MARKET STATE == MS. No trade at 0 state . MS==4 is extremely up and 1 is extremely down
        UL=H_YDay_SPY; // level line 1
        LL=L_YDay_SPY;// level line 2
        NL=(UL+LL)/2; // level line 3
        MS_pr=IIf(O_SPY>=UL,4,IIf(O_SPY>=NL AND O_SPY<UL,3,IIf(O_SPY<NL AND O_SPY>=LL,2,IIf(O_SPY<LL,1,0))));//***
        printf("MS_pr:"+"\t"+MS_pr+"\n");

##### Expected price, short market state
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

##### long_state_price

        B=(HHV(H_5min_SPY,12) + HHV(H_1Min_SPY-L_1Min_SPY,60)+LLV(L_5Min_SPY,12) + LLV(H_1Min_SPY-L_1Min_SPY,60))/2;
        upside_state_price=ValueWhen(Cross(C_5Min_SPY,B),O_SPY);
        downside_state_price=ValueWhen(Cross(B,C_5Min_SPY),O_SPY);
        long_state_price=IIf(C_5Min_SPY>B,upside_state_price,downside_state_price);

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




##### VXX and OPT based sentiment line

        sentiment_vxx=C_VXX;
        sentimetn_pc=C_OP_put/C_OP_call;

        sentiment_vxx_ma=MA(sentiment_vxx,60);
        sentimetn_pc_ma=MA(sentimetn_pc,360);

        sentiment_vxx_HHV=HHV(sentiment_vxx_ma,360);
        sentiment_vxx_LLV=LLV(sentiment_vxx_ma,360);

        MS_VXX_up=ValueWhen(Cross(sentiment_vxx_LLV,sentiment_vxx),C_SPY);//**
        MS_VXX_dn=ValueWhen(Cross(sentiment_vxx,sentiment_vxx_HHV),C_SPY);//**

        MS_pc_up=ValueWhen(Cross(sentimetn_pc_ma-0.02,sentimetn_pc),C_SPY);//**
        MS_pc_dn=ValueWhen(Cross(sentimetn_pc,sentimetn_pc_ma+0.02),C_SPY);//**


        sentiment_upline=ValueWhen(Cross(O_SPY,MS_VXX_up) OR Cross(O_SPY,MS_pc_up),O_SPY);//Max(MS_VXX_up,MS_pc_up);//green
        sentiment_dnline=ValueWhen(Cross(MS_VXX_dn,O_SPY) OR Cross(MS_pc_dn,O_SPY),O_SPY);//Min(MS_VXX_dn,MS_pc_dn);//red
        sentiment_nl=(sentiment_upline+sentiment_dnline)/2; // used at incl_indicator

##### VELOCITY

when Low_velocity_price_upward=ref(Low_velocity_price_upward,-1) and O_SPY>Low_velocity_price_upward then buy

                velocity=Sum(IIf((O_SPY-Ref(O_SPY,-60))/0.05>1,1,IIf((O_SPY-Ref(O_SPY,-60))/-0.05>1,-1,0)),60);
                high_velocity=HHV(velocity,300);
                low_velocity=LLV(velocity,300);
                High_velocity_price_dnward=ValueWhen(Cross(ValueWhen(velocity<(high_velocity-5),O_SPY),O_SPY),O_SPY);
                Low_velocity_price_upward=ValueWhen(Cross(ValueWhen(velocity>(low_velocity+5),O_SPY),O_SPY),O_SPY);
                printf("High_velocity_price_dnward:"+"\t"+High_velocity_price_dnward+"\n");
                printf("Low_velocity_price_upward:"+"\t"+Low_velocity_price_upward+"\n");
