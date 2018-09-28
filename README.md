# SMAI-NN-CNN-Scratch

# CNN - LeNet

Implementation of forward pass of the neural network in LeNet folder

# CIFAR Dataset

Analyze result variations due to parameter changes: batch-size, activa-
tion function (tanh, sigmoid, relu), learning-rate, number of convolutional-filters and
number of (convolutional) layers.

# 3 Layer Neural Network

Implement a simple 3 layer feed-forward neural network for multi-class classification problem.
Implement back-propagation for training the parameters.

Activation function:​ I have used Relu Activation function and tanh activation function.
● Relu is simple and efficient.
● Relu and tanh are easy to implement as relu is just 2 lines of code and tanh already exist
in numpy.
● Relu helps in dropping some neurons at a time and helps in not overfitting data.
● Tanh is used as its graph is over both positive and negative side.
Stopping Condition:​ loss>delta and delta_loss>0.01* delta and steps 20000, as loss is very
loss or decrease in loss is small during iterations, so it is better to stop here, as accuracy will not
increase that much in some time

# Comparision between Activation functions

![Alt text](NeuralNetwork/compare.png?raw=true "Comparison")
