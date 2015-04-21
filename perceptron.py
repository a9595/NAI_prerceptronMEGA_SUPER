import random

from matplotlib.pyplot import plot, show
from scipy.linalg import norm

from data import generate_data


class Perceptron:
    def __init__(self):
        self.weights = [random.random(), random.random(), random.random()]  # depends on amount of inputs
        # self.training_data_arr = get_training_data()
        # self.input_data = self.training_data_arr[:, :4]  # get fist 4 columns - only user input
        # self.input_results = self.training_data_arr[:, -1]
        self.learning_rate = 0.5  # learning speed
        self.train(generate_data(50))

    def activation(self, inputs_row):
        rez = self.weights[0] * inputs_row[0] + \
              self.weights[1] * inputs_row[1] + \
              self.weights[2] * inputs_row[2]
        if rez >= 0:
            return 1
        else:
            return 0

    def train(self, training_data):
        learned = False
        iteration = 0
        while not learned:
            global_error = 0.0
            for row in training_data:  # for each sample
                response = self.activation(row)
                if row[3] != response:  # if we have a wrong response
                    iter_error = row[3] - response  # desired response - actual response
                    self.update_weights(row, iter_error)
                    global_error += abs(iter_error)
            iteration += 1
            if global_error == 0.0 or iteration >= 100:  # over fitting
                learned = True  # stop learning

    def update_weights(self, training_data_row, iter_error):
        self.weights[0] += self.learning_rate * iter_error * training_data_row[0]
        self.weights[1] += self.learning_rate * iter_error * training_data_row[1]
        self.weights[2] -= self.learning_rate * iter_error


perceptron = Perceptron()  # perceptron instance

test_set = generate_data(100)  # test set generation

# Perceptron test
for x in test_set:
    r = perceptron.activation(x)
    if r != x[3]:  # if the response is not correct
        print('error')
    if r == 1:
        plot(x[0], x[1], 'ob')
    else:
        plot(x[0], x[1], 'or')

# plot of the separation line.
# The separation line is orthogonal to w
n = norm(perceptron.weights)
ww = []
for w in perceptron.weights:
    ww.append(w / n)

ww1 = [ww[1], -ww[0]]
ww2 = [-ww[1], ww[0]]
plot([ww1[0], ww2[0]], [ww1[1], ww2[1]], '--k')
show()