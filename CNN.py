import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Input,Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D,Convolution2D, MaxPooling2D
from keras.optimizers import Adam,SGD
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
import numpy as np
from keras.regularizers import l2,l1

model = Sequential()
model.add(Convolution1D(25, 3, border_mode='valid',input_shape=(, 20)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution1D(50, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.output_shape
model.add(Convolution1D(128, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.output_shape
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))
ad=Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',optimizer=ad,metrics=['accuracy'])
hist=model.fit(x_train, y_train,batch_size=128,nb_epoch=50,validation_data=(x_test, y_test))
