LSTM For Sequence Classification With Dropout

Recurrent Neural networks like LSTM generally have the problem of overfitting. Dropout can be applied between layers using the Dropout Keras layer. We can do this easily by adding new Dropout layers between the Embedding and LSTM layers and the LSTM and Dense output layers. 
https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

The way Keras LSTM layers work is by taking in a numpy array of 3 dimensions (N, W, F) where N is the number of training sequences, W is the sequence length and F is the number of features of each sequence. I chose to go with a sequence length (read window size) of 50 which allows for the network so get glimpses of the shape of the sin wave at each sequence and hence will hopefully teach itself to build up a pattern of the sequences based on the prior window received.

http://colah.github.io/posts/2015-08-Understanding-LSTMs/
