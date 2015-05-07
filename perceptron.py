import random

from matplotlib.pyplot import plot, show
from scipy.linalg import norm


LEARNING_RATE = 0.5

GENERATED_DATA_AMOUNT = 100

ITERATION_MAX = 100


def generate_data(n):
    inputs = []
    for i in range(n):
        bias = -1  # x1.w1 + x2.w2 - 1.w0 = 0
        inputs.append([random.uniform(-1, 0),
                       random.uniform(0, 1),
                       bias,
                       1])

        inputs.append([random.uniform(0, 1),
                       random.uniform(-1, 0),
                       bias,
                       0])
    return inputs


class Perceptron:
    def __init__(self):
        self.weights = []
        self.graph_data_arr = []
        self.error_count = 0
        for index in range(3):
            self.weights.append(random.random())
        # self.weights = [random.random(), random.random(), random.random()]  # depends on amount of inputs
        self.learning_rate = LEARNING_RATE  # learning speed
        self.train_data = generate_data(GENERATED_DATA_AMOUNT)
        self.train()

    def activation(self, inputs_row):
        # dot product calc
        # inputs_row example = [-0.81239265210117029, 0.85921322532026534, -1, 1]
        dot_product = 0
        iteration = 0
        for w, i in zip(self.weights, inputs_row):
            dot_product += w * i
            iteration += 1

        threshold = 0
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
                    self.add_graph_data(data_row[0], data_row[1], my_response)
            iteration += 1
            if global_error == 0 or iteration >= ITERATION_MAX:  # over fitting
                learned = True  # stop learning

    def update_weights(self, training_data_row, iter_error):
        for idx in range(len(self.weights) - 1):
            self.weights[idx] += self.learning_rate * iter_error * training_data_row[idx]  # delta rule
        self.weights[-1] -= self.learning_rate * iter_error  # bias weights change

    def add_graph_data(self, input1, input2, response_param):
        self.graph_data_arr.append([input1, input2, response_param])

    def get_error_percent(self):
        return (self.error_count / ITERATION_MAX) * 100


# ------------------------ Creating a perceptron object and test:


perceptron = Perceptron()  # perceptron instance
perceptron.train()

# Perceptron test
# errors_count = 0
# for row in test_set:
#     response = perceptron.activation(row)
#     if response != row[-1]:  # if the response is not correct
#         # print('error')
#         errors_count += 1
#
#     if response == 1:
#         plot(row[0], row[1], 'ob')
#     else:
#         plot(row[0], row[1], 'og')

for row in perceptron.train_data:
    if row[-1] == 1:
        plot(row[0], row[1], 'ob')
    else:
        plot(row[0], row[1], 'og')


# ====== Console prints:
print("example of generated data: ", perceptron.train_data)
print("error% = ", perceptron.get_error_percent())


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


# http://stackoverflow.com/questions/6554792/whats-the-point-of-the-threshold-in-a-perceptron