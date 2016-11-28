import pandas as pd 
import numpy as np 

import keras
import sys
import glob
from keras.models import Sequential,Graph
from keras.layers import Dense,Activation,Dropout,Flatten
from keras.layers import Convolution1D,MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1,l2
from keras.optimizers import SGD,RMSprop,Adam
from keras.utils import np_utils
import math



data=pd.read_csv(open('input_Data_21aa_angle_psi_cleaned.csv','r'))
data.columns=[0,1,2]

training_data=data.sample(n=4000000,random_state=9877917)
training_ids=training_data[0]

testing_data=data[~data[0].isin(training_ids)].sample(n=400000,random_state=7173)

training_data=training_data.sample(frac=1)

X_Train=training_data[1].apply(lambda x: pd.Series(list(x)))
X_Train=np.reshape(X_Train.as_matrix(), X_Train.shape + (1,))

X_Test=testing_data[1].apply(lambda x: pd.Series(list(x)))
X_Test=np.reshape(X_Test.as_matrix(), X_Test.shape + (1,))

Y_Train=training_data[2]
Y_Train=Y_Train.apply(lambda x: math.cos(x))

Y_Test=testing_data[2]
Y_Test=Y_Test.apply(lambda x: math.cos(x))

print X_Train.shape
print X_Test.shape
print Y_Train.shape
print Y_Test.shape

model=Sequential()
model.add(Convolution1D(128,60,subsample_length=20,activation='relu',input_shape=(420,1)))
model.add(BatchNormalization())
model.add(Convolution1D(256,3,activation='relu'))
model.add(BatchNormalization())
model.add(Convolution1D(384,3,activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='tanh'))
model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])

#model.load_weights('weights_Secondarystructure.hdf5')

checkpointer=keras.callbacks.ModelCheckpoint(filepath='hard_weights_Secondarystructure.hdf5',monitor='val_acc',verbose=1,save_best_only=True)
model.fit(X_Train,Y_Train,callbacks=[checkpointer],shuffle=True,batch_size=128,nb_epoch=25,validation_data=(X_Test,Y_Test))
