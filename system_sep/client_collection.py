import zmq
import datetime
import pandas as pd
import numpy as np
import numpy
from numpy import inf

import json
import plotly_stream as plyst
import plotly.tools as plyt
import plotly.plotly as ply
#!pip install plotly
import tpqib
import datetime

#from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import matplotlib.pyplot as plt

import pickle

iterations = 0
df = pd.DataFrame()
pdf= pd.DataFrame()
final=pd.DataFrame()

port = "7000"

# socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.SUB)

print ("Collecting & plotting stock prices.")
socket.connect("tcp://localhost:%s" % port)

socket.setsockopt_string(zmq.SUBSCRIBE, u'SPY')
