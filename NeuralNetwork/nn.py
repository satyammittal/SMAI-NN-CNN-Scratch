import numpy as np
import pandas as pd
import sys
class NeuralNetwork:
    def __init__(self, inputl, hidden, outputl, fn):
        self.hidden_nodes = hidden
        self.num_features = inputl
        self.num_labels = outputl
        self.act_fn = fn
        np.random.seed(0)
        self.layer1_weights_array = np.random.normal(0, 1, [self.num_features, self.hidden_nodes]) / (1.0 * np.sqrt(self.num_features)) 
        self.layer1_biases_array = np.zeros((1, self.hidden_nodes))
        self.layer2_weights_array = np.random.normal(0, 1, [self.hidden_nodes, self.num_labels]) / (1.0 * np.sqrt(self.hidden_nodes))
        self.layer2_biases_array = np.zeros((1, self.num_labels))

    def relu_activation(self, data_array):
        return np.maximum(data_array, 0)
    
    def tanh_derivate(self, data_array):
        return (1 - np.power(data_array,2))
    
    def tanh(self, output_array):
        return np.tanh(output_array)

    def softmax(self, output_array):
        logits_exp = np.exp(output_array.astype(np.float)-np.max(output_array))
        val = logits_exp / np.nansum(logits_exp, axis = 1, keepdims = True)
        val[np.isnan(val)] = 0
        return val
    
    def cross_entropy_softmax_loss_array(self, softmax_probs_array, y_onehot):
        indices = np.argmax(y_onehot, axis = 1).astype(int)
        predicted_probability = softmax_probs_array[np.arange(len(softmax_probs_array)), indices]
        log_preds = np.log(predicted_probability+sys.float_info.epsilon)
        log_preds[np.isnan(log_preds)] = 0
        loss = (-1.0 * np.nansum(log_preds)) / (1.0 * len(log_preds) )
        return loss
    
    def get_labels_in_one_hot_encoding(self, labels):
        labels_onehot = np.zeros((labels.shape[0], self.num_labels)).astype(int)
        labels_onehot[np.arange(len(labels)), labels.astype(int)] = 1
        return labels_onehot
    

    def regularization_L2_softmax_loss(self, reg_lambda, weight1, weight2):
        weight1_loss = 0.5 * reg_lambda * np.nansum(weight1 * weight1)
        weight2_loss = 0.5 * reg_lambda * np.nansum(weight2 * weight2)
        return weight1_loss + weight2_loss
    
    def train(self, data, labels, reg_lambda, learning_rate):
        pr_loss = 0
        delta_loss = 100
        loss = 100
        delta = 0.0001
        step = 0
        while loss>delta and delta_loss>0.01* delta and step<20000:
            input_layer = np.dot(data, self.layer1_weights_array)
            if self.act_fn == "relu":
                hidden_layer = self.relu_activation(input_layer + self.layer1_biases_array)
            else:
                hidden_layer = self.tanh(input_layer + self.layer1_biases_array)
            output_layer = np.dot(hidden_layer, self.layer2_weights_array) + self.layer2_biases_array
            output_probs = self.softmax(output_layer)
            loss = self.cross_entropy_softmax_loss_array(output_probs, labels)
            loss += self.regularization_L2_softmax_loss(reg_lambda, self.layer1_weights_array, self.layer2_weights_array)

            output_error_signal = (output_probs - labels) / output_probs.shape[0]

            error_signal_hidden = np.dot(output_error_signal, self.layer2_weights_array.T) 
            if self.act_fn == "relu":
                error_signal_hidden[hidden_layer <= 0] = 0
            else:
                error_signal_hidden *= self.tanh_derivate(hidden_layer)
            gradient_layer2_weights = np.dot(hidden_layer.T, output_error_signal)
            gradient_layer2_bias = np.sum(output_error_signal, axis = 0, keepdims = True)

            gradient_layer1_weights = np.dot(data.T, error_signal_hidden)
            gradient_layer1_bias = np.sum(error_signal_hidden, axis = 0, keepdims = True)

            gradient_layer2_weights += reg_lambda * self.layer2_weights_array
            gradient_layer1_weights += reg_lambda * self.layer1_weights_array

            self.layer1_weights_array -= learning_rate * gradient_layer1_weights
            self.layer1_biases_array -= learning_rate * gradient_layer1_bias
            self.layer2_weights_array -= learning_rate * gradient_layer2_weights
            self.layer2_biases_array -= learning_rate * gradient_layer2_bias
            delta_loss = abs(loss-pr_loss)
            pr_loss = loss
            if step % 500 == 0:
                    print 'Loss at step {0}: {1}'.format(step, loss)
            step += 1
        return self.predict(data, labels)

    def predict(self, test_dataset, test_labels):
        input_layer = np.dot(test_dataset, self.layer1_weights_array)
        if self.act_fn == "relu":
            hidden_layer = self.relu_activation(input_layer + self.layer1_biases_array)
        else:
            hidden_layer = self.tanh(input_layer + self.layer1_biases_array)
        scores = np.dot(hidden_layer, self.layer2_weights_array) + self.layer2_biases_array
        probs = self.softmax(scores)
        return self.accuracy(probs, test_labels)
        
    def accuracy(self, predictions, labels):
        preds_correct_boolean =  np.argmax(predictions, 1) == np.argmax(labels, 1)
        correct_predictions = np.sum(preds_correct_boolean)
        accuracy = (100.0 * correct_predictions) / predictions.shape[0]
        return accuracy