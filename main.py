# ============== Perceptron testing
from matplotlib.pyplot import plot, show
from scipy.linalg import norm
from perceptron import Perceptron, generate_data, GENERATE_TEST_SET_AMOUNT, ITERATION_MAX

perceptron = Perceptron()
print("example of generated data: ", perceptron.train_data)

test_set = generate_data(GENERATE_TEST_SET_AMOUNT)  # test set generation

error_count = 0
# Perceptron test
for data_raw in test_set:
    resp = perceptron.activation(data_raw)
    if resp != data_raw[-1]:  # if the response is not correct
        error_count += 1
    if resp == 1:
        plot(data_raw[0], data_raw[1], 'ob')
    else:
        plot(data_raw[0], data_raw[1], 'or')

print("error % =", (error_count / ITERATION_MAX) * 100)

vector_length = norm(perceptron.weights)
normalized_vector_arr = []
for weight in perceptron.weights:
    normalized_vector_arr.append(weight / vector_length)

orthogonal_up = [normalized_vector_arr[1], -normalized_vector_arr[0]]
orthogonal_down = [-normalized_vector_arr[1], normalized_vector_arr[0]]
plot([orthogonal_up[0], orthogonal_down[0]], [orthogonal_up[1], orthogonal_down[1]], '--k')
show()