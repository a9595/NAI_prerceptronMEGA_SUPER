import random

from matplotlib.pyplot import plot, show
from scipy.linalg import norm


LEARNING_RATE = 0.5

GENERATED_DATA_AMOUNT = 100

ITERATION_MAX = 100


def generate_data(n):
    inputs = []
    for i in range(n):
        inputs.append([random.uniform(-1, 0),
                       random.uniform(0, 1),
                       -1,
                       1])  # threshold -1

        inputs.append([random.uniform(0, 1),
                       random.uniform(-1, 0),
                       -1,
                       0])
    return inputs


class Perceptron:
    def __init__(self):
        self.weights = []
        for index in range(3):
            self.weights.append(random.random())
        # self.weights = [random.random(), random.random(), random.random()]  # depends on amount of inputs
        self.learning_rate = LEARNING_RATE  # learning speed
        train_data = generate_data(GENERATED_DATA_AMOUNT)
        self.train(train_data)

    def activation(self, inputs_row):
        # dot product calc
        # inputs_row example = [-0.81239265210117029, 0.85921322532026534, -1, 1]
        dot_product = 0
        for w, i in zip(self.weights, inputs_row):
            dot_product += w * i

        if dot_product >= 0:
            return 1
        else:
            return 0

    def train(self, training_data):
        learned = False
        iteration = 0
        while not learned:
            global_error = 0.0
            for data_row in training_data:  # for each sample
                my_response = self.activation(data_row)
                if data_row[-1] != my_response:  # if we have a wrong response
                    iter_error = data_row[-1] - my_response  # desired response - actual response
                    self.update_weights(data_row, iter_error)
                    global_error += abs(iter_error)  # absolute value
            iteration += 1
            if global_error == 0 or iteration >= ITERATION_MAX:  # over fitting
                learned = True  # stop learning

    def update_weights(self, training_data_row, iter_error):
        for idx in range(len(self.weights) - 1):
            self.weights[idx] += self.learning_rate * iter_error * training_data_row[idx]
        self.weights[-1] -= self.learning_rate * iter_error
        # self.weights[0] += self.learning_rate * iter_error * training_data_row[0]
        # self.weights[1] += self.learning_rate * iter_error * training_data_row[1]
        # self.weights[2] -= self.learning_rate * iter_error


# ------------------------ Creating a perceptron object and test:


perceptron = Perceptron()  # perceptron instance

test_set = generate_data(100)  # test set generation

# Perceptron test
errors_count = 0
for row in test_set:
    response = perceptron.activation(row)
    if response != row[-1]:  # if the response is not correct
        # print('error')
        errors_count += 1

    if response == 1:
        plot(row[0], row[1], 'ob')
    else:
        plot(row[0], row[1], 'og')

# ====== Console prints:
print("example of generated data: ", test_set)
print("error% = ", (errors_count / ITERATION_MAX) * 100)


# plot of the separation line.
# The separation line is orthogonal to w
vector_length = norm(perceptron.weights)
normalized_vector_arr = []
for weight in perceptron.weights:
    normalized_vector_arr.append(weight / vector_length)

orthogonal_up = [normalized_vector_arr[1], -normalized_vector_arr[0]]
orthogonal_down = [-normalized_vector_arr[1], normalized_vector_arr[0]]
plot([orthogonal_up[0], orthogonal_down[0]], [orthogonal_up[1], orthogonal_down[1]], '--k')
show()