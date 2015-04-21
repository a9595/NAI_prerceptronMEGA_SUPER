import random
import numpy as np
from data import get_training_data, generate_data


class Perceptron:
    def __init__(self):
        self.weights = [random.random(), random.random(), random.random()]  # depends on amount of inputs
        # self.training_data_arr = get_training_data()
        # self.input_data = self.training_data_arr[:, :4]  # get fist 4 columns - only user input
        # self.input_results = self.training_data_arr[:, -1]
        self.learning_rate = 0.5  # learning speed
        self.train(generate_data(50))

    def activation(self, inputs_row):
        rez = np.dot(self.weights, inputs_row)
        if rez >= 0:
            return 1
        else:
            return 0

    def train(self, data):
        learned = False
        iteration = 0
        while not learned:
            global_error = 0.0
            for x in data:  # for each sample
                r = self.activation(x)
                if x[2] != r:  # if we have a wrong response
                    iter_error = x[2] - r  # desired response - actual response
                    self.update_weights(x, iter_error)
                    global_error += abs(iter_error)
            iteration += 1
            if global_error == 0.0 or iteration >= 100:  # over fitting
                learned = True  # stop learning

    def update_weights(self, x, iter_error):
        self.weights[0] += self.learning_rate * iter_error * x[0]
        self.weights[1] += self.learning_rate * iter_error * x[1]
        self.weights[2] -= self.learning_rate * iter_error