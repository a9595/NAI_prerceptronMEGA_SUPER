import random
import numpy as np
from data import get_training_data


class Perceptron:
    def __init__(self):
        self.weights = [random.random(), random.random(), random.random(), random.random(),
                        random.random()]  # depends on amount of inputs
        self.training_data_arr = get_training_data()
        self.input_data = self.training_data_arr[:, :4]  # get fist 4 columns - only user input
        self.input_results = self.training_data_arr[:, -1]
        self.learning_rate = 0.5  # learning speed

    def activation(self, inputs_row):
        inputs_row.append(-1)
        rez = np.dot(self.weights, inputs_row)
        if rez >= 0:
            return 1
        else:
            return 0

    def training(self, inputs_row, out):
        while self.activation(inputs_row) != out:
            for r in range(5):
                self.weights[r] -= self.learning_rate * inputs_row[r]  # correction by delta rule