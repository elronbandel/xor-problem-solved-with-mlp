import loglinear as ll
import mlp1 as mlp
import random
import numpy as np
from xor_data import data

STUDENT={'name': 'bandele_passove1',
         'ID': '308130038_305930265'}

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # Compute the accuracy (a scalar) of the current parameters
        prediction = mlp.predict(features, params)
        if prediction == label:
            good += 1
        else:
            bad += 1
        # accuracy is (correct_predictions / all_predictions)
    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        for label, features in train_data:
            loss, grads = mlp.loss_and_gradients(features, label, params)
            cum_loss += loss
            params = [param - grad for param,grad in zip(params, [grad * learning_rate for grad in grads])]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print((I, train_loss, train_accuracy, dev_accuracy))
    return params

if __name__ == '__main__':
    in_dim = 2
    out_dim = 2
    np.random.seed(5)
    train_data = data
    dev_data = data
    num_iterations = 8
    learning_rate = 0.5
    params = mlp.create_classifier(in_dim, 4, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

