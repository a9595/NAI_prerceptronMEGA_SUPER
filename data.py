import random


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