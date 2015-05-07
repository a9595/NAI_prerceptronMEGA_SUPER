import random

from data import generate_data

GENERATE_TEST_SET_AMOUNT = 100
GENERATED_TRAINING_DATA_AMOUNT = 100
LEARNING_RATE = 0.5
ITERATION_MAX = 100


class Perceptron:
    def __init__(self):
        self.weights = []
        for index in range(3):
            self.weights.append(random.random())
        # self.weights = [random.random(), random.random(), random.random()]  # depends on amount of inputs
        self.learning_rate = LEARNING_RATE  # learning speed
        self.train_data = generate_data(GENERATED_TRAINING_DATA_AMOUNT)
        self.train()

    def activation(self, inputs_row):
        # dot product calc
        # inputs_row example = [-0.81239265210117029, 0.85921322532026534, -1, 1]
        dot_product = 0
        for w, i in zip(self.weights, inputs_row):
            dot_product += w * i

        threshold = 0  # x1.w1 + x2.w2 - 1.TH = 0
        if dot_product >= threshold:
            return 1
        else:
            return 0

    def train(self):
        learned = False
        iteration = 0
        while not learned:
            global_error = 0.0
            for data_row in self.train_data:  # for each sample
                my_response = self.activation(data_row)  # is data fired(1)

                if data_row[-1] != my_response:  # if we have a wrong response
                    iter_error = data_row[-1] - my_response  # desired response - actual response
                    self.update_weights(data_row, iter_error)
                    global_error += abs(iter_error)  # absolute value
            iteration += 1
            if global_error == 0 or iteration >= ITERATION_MAX:  # over fitting
                learned = True  # stop

    def update_weights(self, training_data_row, iter_error):
        for idx in range(len(self.weights) - 1):
            self.weights[idx] += self.learning_rate * iter_error * training_data_row[idx]  # delta rule
        self.weights[-1] -= self.learning_rate * iter_error  # bias weights change

        # http://stackoverflow.com/questions/6554792/whats-the-point-of-the-threshold-in-a-perceptron