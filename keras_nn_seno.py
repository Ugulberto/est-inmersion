import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

"""
Este código entrena una red neuronal para identificar los parámetros de una función senoidal a partir de sus valores en 50 puntos.

Primero, se generan miles de funciones senoidales aleatorias del tipo:

f(x)=A⋅sin(Bx+C)
donde los parámetros A, B y C (amplitud, frecuencia y fase) se eligen aleatoriamente dentro de un rango.

A cada función se le calculan 50 valores de salida (f(x)), que se usan como entrada para entrenar la red neuronal. El objetivo del modelo es aprender a predecir los valores de A, B, C solo a partir de esos 50 valores de f(x).

Una vez entrenado, el modelo se pone a prueba con una nueva función senoidal, y se comparan los parámetros reales con los predichos. Además, se grafica la función original y la reconstruida con los parámetros predichos, para ver visualmente la efectividad del entrenamiento de la red neuronal.

NUM_POINTS: El número de valores de la función a predecir que se le van a pasar a la red neuronal.
NUM_EXAMPLES: Número de ejemplos a través de los cuales la red entrena.
EPOCHS: Número de vueltas a todo el dataset que efectúa la red neuronal para entrenarse (más vueltas conllevaría mejores resultados, pero un número demasiado alto de las mismas daría lugar al sobreentrenamiento (se perdería capacidad de predicción)).
BATCH_SIZE: cantidad de ejemplos del conjunto de datos que se usan en cada paso del entrenamiento de la red neuronal. Pasar todos los datos sería ineficiente, pasar sólo uno sería demasiado impreciso. Por ello se ilegen mini-batches basados en las potencias de dos (de tamaño 32, 64, 128).
"""


# -------------------------------
#   Configuración del problema
# -------------------------------
NUM_POINTS = 50      # Número de puntos por función
NUM_EXAMPLES = 50000  # Tamaño del dataset (generado de manera automática)
EPOCHS = 100
BATCH_SIZE = 64

# -------------------------------
#      Generación de datos
# -------------------------------
def crear_funcion_senoidal():
    amplitud = np.random.uniform(0.5, 2.0)
    frecuencia = np.random.uniform(0.5, 3.0)
    fase = np.random.uniform(-np.pi, np.pi)
    
    x_vals = np.linspace(0, 2 * np.pi, NUM_POINTS)
    y_vals = amplitud * np.sin(frecuencia * x_vals + fase)
    
    return y_vals, np.array([amplitud, frecuencia, fase])

def generar_dataset(n_ejemplos):
    entradas = []
    parametros = []
    for _ in range(n_ejemplos):
        y, p = crear_funcion_senoidal()
        entradas.append(y)
        parametros.append(p)
    return np.array(entradas), np.array(parametros)

print("Generando datos...")
X, Y = generar_dataset(NUM_EXAMPLES)

# Escalamos las salidas para facilitar el aprendizaje
scaler = StandardScaler()
Y_escalado = scaler.fit_transform(Y)

# -------------------------------
#     Definición del modelo
# -------------------------------
modelo = keras.Sequential([
    layers.Input(shape=(NUM_POINTS,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(3)  # Salida: A, B, C
])

modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])

# -------------------------------
#         Entrenamiento
# -------------------------------
print("Entrenando el modelo...")

historial = modelo.fit(
    X, Y_escalado,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# -------------------------------
# Evaluación con un nuevo ejemplo
# -------------------------------
print("\nProbando con un nuevo ejemplo...")

x_eval = np.linspace(0, 2 * np.pi, NUM_POINTS)
y_real, params_reales = crear_funcion_senoidal()

# Redimensionar para la red
y_entrada = y_real.reshape(1, -1)

# Predicción y desescalado
prediccion_escalada = modelo.predict(y_entrada)
params_predichos = scaler.inverse_transform(prediccion_escalada)[0]

# -------------------------------
#          Resultados
# -------------------------------
A_real, B_real, C_real = params_reales
A_pred, B_pred, C_pred = params_predichos
error_abs = np.abs(params_predichos - params_reales)

print("\nResultado del test:")
print(f"  A real      = {A_real:.4f}    | A predicho      = {A_pred:.4f}    | error = {error_abs[0]:.4f}")
print(f"  B real      = {B_real:.4f}    | B predicho      = {B_pred:.4f}    | error = {error_abs[1]:.4f}")
print(f"  C real      = {C_real:.4f}    | C predicho      = {C_pred:.4f}    | error = {error_abs[2]:.4f}")

# -------------------------------
#      Gráfico comparativo
# -------------------------------
y_predicha = A_pred * np.sin(B_pred * x_eval + C_pred)

plt.figure(figsize=(8, 4))
plt.plot(x_eval, y_real, label="Función original")
plt.plot(x_eval, y_predicha, '--', label="Reconstrucción")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Comparación entre función original y predicha")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
