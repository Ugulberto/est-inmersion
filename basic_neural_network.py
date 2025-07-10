import numpy as np # type: ignore

# Definimos las capas, los pesos, los sesgos y los valores a predecir
input_layer = [1.34, 1.65]
weights = [
    [1.67, 1.345],
    [-1.56, 0]
]
bias = [0.0]
targets = np.array([1, 0])

# Ahora hacemos un producto escalar
dot_product_1 = np.dot(input_layer, weights[0])
dot_product_2 = np.dot(input_layer, weights[1])

print(f"El resultado del producto escalar con la primera capa de pesos es: {dot_product_1}")
print(f"El resultado del producto escalar con la segunda capa de pesos es: {dot_product_2}")

sigmoid = lambda x: 1 / (1 + np.exp(x))
sigmoid_derivative = lambda x: sigmoid(x) * (1 - sigmoid(x))

def predict(input_layer, weights_layer, bias):
    layer_1 = np.dot(input_layer, weights_layer) + bias # bias o sesgos son los valores que se suman al producto escalar para ajustar la funcion de activacion por encima del origen
    layer_2 = sigmoid(layer_1) # Usamos la sigmoide como función de activación si la salida es 0 o 1, no es el caso propuesto en este trabajo pero sirve como ejemplo
    return layer_2

predictions = np.array([
    predict(input_layer, weights[0], bias),
    predict(input_layer, weights[1], bias)
])

print(f"Las predicciones son: {predictions}") # Esas predicciones serían incorrectas en la mayoría de los casos 

errors = np.zeros(len(predictions))

# Por ende debemos entrenar la red para obtener resultados precisos
for index in range(len(predictions)):
    
    prediction = predictions[index]
    target = targets[index]

    mean_squared_error = np.square(prediction - target)
    print(f"Predicción: {prediction}, objetivo: {target}, error: {mean_squared_error}")

    errors[index] = mean_squared_error[0]

    # Este error cuadrático nos indica cómo de alejada está la predicción del objetivo. Al estar definido por una función cuadrática, para saber si debemos incrementar o reducir los pesos tenemos que derivar este error.
    derivative = 2 * (prediction - target)
    print(f"La derivada es: {derivative}")

    weights[index] += derivative
    predictions[index] = predict(input_layer, weights[index], bias)

print(predictions)

errors_2 = np.zeros(len(predictions))

for index in range(len(predictions)):
    prediction = predictions[index]
    target = targets[index]
    mean_squared_error = np.square(prediction - target)

    errors_2[index] = mean_squared_error[0]

print(predictions)

print(f"Los errores han pasado de {errors} a {errors_2}")

for index in range(len(predictions)):
    prediction = predictions[index]
    target = targets[index]
    weights_layer = weights[index]
    derror_dprediction = 2 * (prediction - target)
    layer_1 = np.dot(input_layer, weights_layer) + bias
    dprediction_dlayer1 = sigmoid_derivative(layer_1)
    dlayer1_dbias = 1

    derror_dprediction = (
        derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
    )