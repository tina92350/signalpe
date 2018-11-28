#!/usr/bin/env python3
import numpy as np
from keras import utils

proteins = 'GAVLIFWYDHNEKQMRSTCP'
classification = {
        'S': 0,
        't': 1,
        'T': 1,
        '.': 2,
        }

def build_train_data(dataset):

    seq = np.zeros((len(dataset)//3, 96, 21))
    label = np.zeros((len(dataset)//3, 96, 3))

    for i in range(0, len(dataset), 3):
        _x = dataset[i+1].decode('utf-8').replace('\n', '')
        _y = dataset[i+2].decode('utf-8').replace('\n', '')
      
        for j, x in enumerate(_x[:96]):
           seq[i//3][j][proteins.index(x) if x in proteins else 20] = 1

        for k, y in enumerate(_y[:96]):
           label[i//3][k][classification[y]] = 1

    return seq, label

if '__main__' == __name__:
    with open('./data/train/train.fasta', 'rb') as f:
        train = f.readlines()
    
    data_x, data_y = build_train_data(train)

    print('shape of data_x:', data_x.shape)
    print('shape of data_y:', data_y.shape)

    np.save('data_x.npy', data_x)
    np.save('data_y.npy', data_y)
