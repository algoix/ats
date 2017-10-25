##### Expected line
1. price 12 sec back =P_12
2. Expected move up or down. If present price below 1m line then down. Vice varsa.
3. Expected change in 5m == 0.30. in 12 sec 0.01
4. Present expected price P_12+-0.01.

        expected_price=IIf(MA(O_SPY,12)>C_1Min_SPY,C_1Min_SPY+0.01,C_1Min_SPY-0.01);

##### Expected market line


##### sentiment_upline and sentiment_dnline

