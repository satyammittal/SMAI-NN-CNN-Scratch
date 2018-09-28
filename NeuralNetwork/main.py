import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from nn import NeuralNetwork


def merge_arrays(arr):
    out = np.array([])
    #print out.shape
    for r in arr:
        if r.shape[0]==0:
            pass
        elif out.shape[0]==0:
            out = r
        else:
            out = np.concatenate((out, r), axis=0)
    #print out.shape
    return out

def one_hot_encoding(labels, num_labels):
    labels_onehot = np.zeros((labels.shape[0], num_labels)).astype(int)
    labels_onehot[np.arange(len(labels)), labels.astype(int)] = 1
    return labels_onehot

def cross_validate(hidden_layer, k, split_arr, activation):
  test_cross = []
  train_cross = []
  for j in range(k):
      test_arr = split_arr[j]
      train_arr = merge_arrays(split_arr[:j])
      t = merge_arrays(split_arr[j+1:])
      train_arr = merge_arrays([train_arr,t])
      train_x = train_arr[:,:-1]
      train_y = one_hot_encoding(train_arr[:,-1], 3)
      test_x = test_arr[:,:-1]
      test_y = one_hot_encoding(test_arr[:,-1], 3)
      network = NeuralNetwork(train_x.shape[1],hidden_layer,train_y.shape[1], activation)
      train_cross.append(network.train(train_x,train_y,0.01, 0.05))
      test_cross.append(network.predict(test_x,test_y))
  return np.array(train_cross).mean(), np.array(test_cross).mean()

def read_dataset(loc):
    df=pd.read_csv(loc, sep=',',header=None)
    df = df[((df[16]==0) | (df[16]==1) | (df[16]==2) | (df[16]==3))]
    dataset = df.values.astype('float')
    #np.random.shuffle(dataset)
    return dataset

def NN(train_x, train_y, test_x, test_y, hidden_layer, activation):
    network = NeuralNetwork(train_x.shape[1],hidden_layer,train_y.shape[1], activation)
    a = network.train(train_x,train_y,0.01, 0.001)
    b = network.predict(test_x, test_y)
    return a, b

def dermatology():
    df=pd.read_csv('dermatology.csv', sep=',',header=None)
    df = df[((df[34]==1) | (df[34]==2) | (df[34]==3)) & (df[33]!='?')]
    df[34]-=1
    dataset = df.values.astype('float') 
    k = 5
    r = np.copy(dataset)
    np.random.shuffle(r)
    split_arr = np.asarray(np.array_split(r, k))
    print("dermatology Relu" + str(cross_validate(10, k, split_arr, "relu")))
    print("dermatology Tanh" + str(cross_validate(10, k, split_arr, "tanh")))

def pendigits():
    train = read_dataset('pendigits.tes')
    test = read_dataset('pendigits.tes')
    train_in = train[:,:-1]
    train_out = one_hot_encoding(train[:,-1], 4)
    test_in = test[:,:-1]
    test_out = one_hot_encoding(test[:,-1],4 )
    print("pendigits Relu" + str(NN(train_in, train_out, test_in, test_out, 5, "relu")))

dermatology()
pendigits()