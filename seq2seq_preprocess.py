import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

def preproc(part):
    data_new=pd.read_csv('/home/samskruthi/sscode_hard_indexadded.csv')
    data=data_new.sample(frac=1)
    part_length=part
    data1=data['0'].str.replace('A','10000000000000000000 ')
    data1=data1.str.replace('C','01000000000000000000 ')
    data1=data1.str.replace('D','00100000000000000000 ')
    data1=data1.str.replace('E','00010000000000000000 ')
    data1=data1.str.replace('F','00001000000000000000 ')
    data1=data1.str.replace('G','00000100000000000000 ')
    data1=data1.str.replace('H','00000010000000000000 ')
    data1=data1.str.replace('I','00000001000000000000 ')
    data1=data1.str.replace('K','00000000100000000000 ')
    data1=data1.str.replace('L','00000000010000000000 ')
    data1=data1.str.replace('M','00000000001000000000 ')
    data1=data1.str.replace('N','00000000000100000000 ')
    data1=data1.str.replace('P','00000000000010000000 ')
    data1=data1.str.replace('Q','00000000000001000000 ')
    data1=data1.str.replace('R','00000000000000100000 ')
    data1=data1.str.replace('S','00000000000000010000 ')
    data1=data1.str.replace('T','00000000000000001000 ')
    data1=data1.str.replace('W','00000000000000000100 ')
    data1=data1.str.replace('V','00000000000000000010 ')
    data1=data1.str.replace('Y','00000000000000000001 ')
    data2=data1.str.strip()
    data_amino=data2.str.split(' ')
    data3=data['1'].str.replace('B','100 ')
    data3=data3.str.replace('C','010 ')
    data3=data3.str.replace('H','001 ')
    data4=data3.str.strip()
    data_struc=data4.str.split(' ')
    amino_row=[]
    for i in range(len(data_amino)):
        length,parts_no=len(data_amino[i]),(len(data_amino[i])//part_length)+1
        for j in range(0,parts_no):
            if (j+1)*part_length<length:
                amino_row.append((data_amino[i][part_length*j:part_length*j+part_length]))
            else:
                amino_row.append((data_amino[i][part_length*j:]))
    amino_row1=[x for x in amino_row if x != []]
    struc_row=[]
    for i in range(len(data_struc)):
        length,parts_no=len(data_struc[i]),(len(data_struc[i])//part_length)+1
        for j in range(0,parts_no):
            if (j+1)*part_length<length:
                struc_row.append((data_struc[i][part_length*j:part_length*j+part_length]))
            else:
                struc_row.append((data_struc[i][part_length*j:]))
    struc_row1=[x for x in struc_row if x != []]
    data_x=[]
    for i in amino_row1:
        each_row=[]
        for j in i:
            each_row.append(list(j))
        data_x.append(each_row)
    data_y=[]
    for i in struc_row1:
        each_row=[]
        for j in i:
            each_row.append(list(j))
        data_y.append(each_row)
    x=pad_sequences(data_x).astype('float32')
    y=pad_sequences(data_y).astype('float32')
    print x.shape
    print y.shape
    indices = np.arange(len(x))
    split=len(x)*9//10
    train_indices=np.random.permutation(split)
    test_indices=indices[split:]
    x_train=x[train_indices]
    x_test=x[test_indices]
    y_train=y[train_indices]
    y_test=y[test_indices]
    print x_train.shape,x_test.shape
    print y_train.shape,y_test.shape
    return x_train,y_train,x_test,y_test


part_length=500
x_train,y_train,x_test,y_test=preproc(part_length)

x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
y_train=y_train.astype('float32')
y_test=y_test.astype('float32')
