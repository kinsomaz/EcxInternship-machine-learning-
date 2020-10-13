# %% [code]
#Explanatory Data Analysis
#I observed from the dataset that the production of the candy production company was given month by month from 1972 to 2017
#I  also predicted the production Month by Month for the following years.... Running the code frequently tends to give you the Analysis of the following year
#I could not predict the production for the next 4 years at once because the dataset was not enough to make the analysis accurate
#The model i used was the Encoder-Decoder model under the Multi-Step LSTM Models
#I split the dataset into a 80-20 split for the train and test data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import random
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn.metrics import mean_absolute_error

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
#read the csv file
data = pd.read_csv("C:\\Users\\master\\Downloads\\candy_production.csv")
t =  data.iloc[: , [ 1]] 
#define the input sequence
raw_seq = t.values
raw_seq_train = raw_seq[:428]

# choose a number of time steps
n_steps_in, n_steps_out = 12, 12
# split into samples
X, y = split_sequence(raw_seq_train, n_steps_in, n_steps_out)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
y = y.reshape((y.shape[0], y.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(100, activation='relu',return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=100, verbose=0)
# demonstrate prediction
raw_seq_test = raw_seq[428:]
x = raw_seq_test.reshape((10 , 12 , raw_seq.shape[1] ))
r = random.randint(0 , 8)
x_input = x[r]

x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

#Calculating the Mean Absolute Error(MAE) and Mean Percentage Error(MPE)
expected = x[r + 1]
predictions = yhat.reshape((12,1))
mae = mean_absolute_error(expected, predictions)
print('MAE: %f' % mae)

def mean_absolute_percentage_error(expected, predictions): 
    y_true, y_pred = np.array(expected), np.array(predictions)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(expected, predictions)
print('MAPE: %f' % mape)
