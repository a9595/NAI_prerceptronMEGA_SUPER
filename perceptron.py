import random
import numpy as np
from data import get_training_data, generate_data


class Perceptron:
    def __init__(self):
        self.weights = [random.random(), random.random(), random.random()]  # depends on amount of inputs
        # self.training_data_arr = get_training_data()
        # self.input_data = self.training_data_arr[:, :4]  # get fist 4 columns - only user input
        # self.input_results = self.training_data_arr[:, -1]
        self.inputs = generate_data(50)
        self.learning_rate = 0.5  # learning speed
        self.training_all_data()

    def activation(self, inputs_row):
        inputs_row.append(-1)
        rez = np.dot(self.weights, inputs_row)
        if rez >= 0:
            return 1
        else:
            return 0

    def training(self, inputs_row):
        while self.activation(inputs_row) != inputs_row[3]:
            for r in range(2):
                self.weights[r] -= self.learning_rate * inputs_row[r]  # correction by delta rule

    def training_all_data(self):
        for data in self.inputs:
            self.training(data)