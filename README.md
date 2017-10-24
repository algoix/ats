# Python ML input to Amibroker
https://github.com/algoix/Quant_Trade/blob/12f8ad4b068aa83c16f2d6d84d098792731baaaf/ML.py

                                  price        pREG        pSVR       class  \
    2017-09-06 20:51:10.712978  246.869785  246.871884  246.867284    0.0   

                                    km      LSTM  
    2017-09-06 20:51:10.712978  246.856048 -0.023423  

This output is from python ML pipeline. This output is saved in hard disk ( dropbox) to be used by Amibroker
      
      output.tail(1).to_csv('/home/octo/Dropbox/ml_output.txt', sep=',', encoding='utf-8')

A part of AFL shows to import the ML output 
  
    // open file
    fh = fopen("C:\\Users\\Michal\\Dropbox\\ml_output.txt", "r" );

    if( fh )
    {
        i = 0;
        ...
https://github.com/algoix/Quant_Trade/blob/L1/incl_import.afl        
    
