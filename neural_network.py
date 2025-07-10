import matplotlib.pyplot as plt
import numpy as np

class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))
    
    def predict(self, input_layer):
        layer_1 = np.dot(input_layer, self.weights) + self.bias
        prediction = self._sigmoid(layer_1)
        return prediction

    def _compute_gradients(self, input_layer, target):
        layer_1 = np.dot(input_layer, self.weights) + self.bias
        prediction = self._sigmoid(layer_1)
        
        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_derivative(layer_1)

        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_layer)

        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            if current_iteration % 100 == 0:
                cumulative_error = 0
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors

neural_network = NeuralNetwork(0.1)

input_layer = [1.456, 1.23]

prediction = neural_network.predict(input_layer)

input_vectors = np.array(
   [
       [3, 1.5],
       [2, 1],
       [4, 1.5],
       [3, 4],
       [3.5, 0.5],
       [2, 0.5],
       [5.5, 1],
       [1, 1],
   ]
)

targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])

learning_rate = 0.1

neural_network = NeuralNetwork(learning_rate)

training_error = neural_network.train(input_vectors, targets, 10000)

plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("cumulative_error.png")

print(neural_network.weights)