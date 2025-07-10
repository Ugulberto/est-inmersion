import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, learning_rate):
        self.weights = [
            np.random.randn(input_dim, 32) * np.sqrt(2 / input_dim),
            np.random.randn(32, 16) * np.sqrt(2 / 32),
            np.random.randn(16, 4) * np.sqrt(2 / 16)
        ]
        self.bias = [
            np.zeros((1, 32)),
            np.zeros((1, 16)),
            np.zeros((1, 4))
        ]
        self.learning_rate = learning_rate

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return (x > 0).astype(float)

    def predict(self, input_layer):
        self.z = []
        self.a = [input_layer]

        curr_layer = input_layer
        for i in range(len(self.weights) - 1):
            print(curr_layer.shape, self.weights[i].shape)
            pre_act = curr_layer @ self.weights[i] + self.bias[i]
            self.z.append(pre_act)
            curr_layer = self._relu(pre_act)
            self.a.append(curr_layer)

        pre_act_final = curr_layer @ self.weights[-1] + self.bias[-1]
        self.z.append(pre_act_final)
        self.a.append(pre_act_final)  # sin activación en la última capa

        return pre_act_final

    def backprop(self, y_true):
        dL_dyhat = (self.a[-1] - y_true)  # derivada error salida

        delta3 = dL_dyhat
        dW3 = self.a[2].T @ delta3
        db3 = delta3

        dL_da2 = delta3 @ self.weights[2].T
        da2_dz2 = self._relu_derivative(self.z[1])
        delta2 = dL_da2 * da2_dz2

        dW2 = self.a[1].T @ delta2
        db2 = delta2

        dL_da1 = delta2 @ self.weights[1].T
        da1_dz1 = self._relu_derivative(self.z[0])
        delta1 = dL_da1 * da1_dz1

        dW1 = self.a[0].T @ delta1
        db1 = delta1

        self.weights[2] -= self.learning_rate * dW3
        self.bias[2] -= self.learning_rate * db3
        self.weights[1] -= self.learning_rate * dW2
        self.bias[1] -= self.learning_rate * db2
        self.weights[0] -= self.learning_rate * dW1
        self.bias[0] -= self.learning_rate * db1

        return [dW1, dW2, dW3], [db1, db2, db3]


def generar_ejemplo(num_puntos=50):
    A = np.random.uniform(0.5, 2.0)
    omega = np.random.uniform(0.5, 2.0)
    phi = np.random.uniform(0, 2*np.pi)
    b = np.random.uniform(-1, 1)

    x = np.linspace(0, 2*np.pi, num_puntos)            # (50,)
    fx = A * np.sin(omega * x + phi) + b               # (50,)

    input_vector = np.vstack([x, fx])                   # (2, 50)
    input_vector = input_vector.flatten().reshape(1, -1) # (1, 100)

    target = np.array([[A, omega, phi, b]])              # (1, 4)
    return input_vector, target





# Entrenamiento
nn = NeuralNetwork(input_dim=100, learning_rate=0.001)

epochs = 2000
losses = []

for epoch in range(epochs):
    input_vec, target_vec = generar_ejemplo()
    y_pred = nn.predict(input_vec)
    nn.backprop(target_vec)
    loss = np.mean((y_pred - target_vec) ** 2)
    losses.append(loss)



num_puntos = 50
x = np.linspace(0, 2 * np.pi, num_puntos).reshape(-1, 1)   # Vector columna (50,1)
input_vector = x.flatten().reshape(1, -1)                  # Forma (1, 50), para input a la red
target_vector = np.sin(x).flatten().reshape(1, -1)         # Salida esperada, (1, 50)

print(nn.predict(input_vector))
