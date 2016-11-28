
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Input,Dense, Dropout, Activation, Flatten,LSTM
from keras.layers import Embedding,merge,TimeDistributed
from keras.layers import Convolution1D, MaxPooling1D
from keras.optimizers import Adam,SGD
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
import numpy as np
from keras.regularizers import l2,l1
from keras.engine import Model

hidden=50
hidden1=50
p=0.25
a=Input(shape=(part_length,20))

rnn_fwd1 = LSTM(hidden, return_sequences=True)(a)
d_fwd1=Dropout(p)(rnn_fwd1)
rnn_bwd1 = LSTM(hidden, return_sequences=True, go_backwards=True)(a)
d_bwd1=Dropout(p)(rnn_bwd1)
rnn_bidir1 = merge([d_fwd1, d_bwd1], mode='sum')

rnn_fwd2 = LSTM(hidden1, return_sequences=True)(rnn_bidir1)
d_fwd2=Dropout(p)(rnn_fwd2)
rnn_bwd2 = LSTM(hidden1, return_sequences=True, go_backwards=True)(rnn_bidir1)
d_bwd2=Dropout(p)(rnn_bwd2)
rnn_bidir2 = merge([d_fwd2, d_bwd2], mode='sum')

rnn_fwd3 = LSTM(hidden1, return_sequences=True)(rnn_bidir2)
d_fwd3=Dropout(p)(rnn_fwd3)
rnn_bwd3 = LSTM(hidden1, return_sequences=True, go_backwards=True)(rnn_bidir2)
d_bwd3=Dropout(p)(rnn_bwd3)
rnn_bidir3 = merge([d_fwd3, d_bwd3], mode='sum')

dense_1= TimeDistributed(Dense(100, activation='tanh'))(rnn_bidir3)
d_dense_1=Dropout(p)(dense_1)

dense_2=TimeDistributed(Dense(100, activation='tanh'))(d_dense_1)
d_dense_2=Dropout(p)(dense_2)

predictions = TimeDistributed(Dense(3, activation='softmax'))(d_dense_2)

model = Model(input=a, output=predictions)
#ad=Adam(lr=0.01)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train,batch_size=128,nb_epoch=50,validation_data=(x_test, y_test))

