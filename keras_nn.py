import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

# Generador de ejemplos
def generar_ejemplo(num_puntos=50):
    A = np.random.uniform(0.5, 2.0)
    omega = np.random.uniform(0.5, 2.0)
    phi = np.random.uniform(0, 2*np.pi)
    b = np.random.uniform(-1, 1)

    x = np.linspace(0, 2*np.pi, num_puntos)
    fx = A * np.sin(omega * x + phi) + b

    # Input: concatenamos x y f(x), forma (100,)
    input_vec = np.concatenate([x, fx])
    # Target: los 4 parámetros
    target_vec = np.array([A, omega, phi, b])
    return input_vec.astype(np.float32), target_vec.astype(np.float32)

# Modelo simple en Keras
model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(100,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(4)  # salida: 4 parámetros
])

model.compile(optimizer=optimizers.Adam(0.001),
              loss=losses.MeanSquaredError())

# Entrenamiento
epochs = 2000
for epoch in range(epochs):
    input_vec, target_vec = generar_ejemplo()
    input_vec = input_vec.reshape(1, -1)  # batch size = 1
    target_vec = target_vec.reshape(1, -1)

    loss = model.train_on_batch(input_vec, target_vec)

    if epoch % 200 == 0:
        print(f'Epoch {epoch}, Loss {loss:.5f}')

# Probar con un ejemplo fijo (función seno simple)
num_puntos = 50
x = np.linspace(0, 2*np.pi, num_puntos)
fx = np.sin(x)
input_test = np.concatenate([x, fx]).reshape(1, -1).astype(np.float32)
pred_params = model.predict(input_test)
print("Predicción parámetros:", pred_params)
