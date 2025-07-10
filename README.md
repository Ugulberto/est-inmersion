# Estancias de inmersión
Este repositorio contiene numerosos códigos .py relacionados con el campo de la inteligencia artificial y las ondas gravitacionales. En este repositorio hay 5 archivos .py:
- basic_neural_network.py es una red neuronal de la forma más básica posible, sin POO, sin módulos... Simplemente se usa numpy para facilitar la manipulación de las listas necesarias para poner en funcionamiento la red. Este archivo contiene una red de decisión y usa la función de activación sigmoide para ello.
- neural_network.py cra una red neuronal sin módulos utilizando POO y la función sigmoide como función de activación, creando una gráfica que explica el error en las predicciones a lo largo de todo el entrenamiento de la red.

Estos dos archivos son breves introducciones a lo que será el verdadero proyecto de práctica de este repositorio: el cual se encuentra en el archivo keras_nn_seno.py. Los dos archivos restantes han sido pruebas que he ido realizando para ir acomodándome con los módulos que se suelen utilizar para la cración de red neuronales, como tensorflow y su herramienta keras.
En este archivo, keras_nn_seno.py, se intenta predecir los parámetros propios de una función senoidal:
```math
f(x)=A⋅sin(Bx+C)
```
A partir de una serie de valores de x y f(x) de dicha función.
En este código se encuentra una explicación de los distintos parámetros que se pueden modificar para ver cómo cambia el funcionamiento de la red neuronal.
Los resultados obtenidos son bastante precisos, y se pueden comprobar gracias a la creación de una gráfica que compara las ecuaciones predecida y original en el ejemplo que se usa como predicción.

# ¿Cómo probar estos scripts?
1. Instala la herramienta git en tu ordenador a través del siguiente enlace: https://git-scm.com/downloads
2. En tu terminal de comandos, en el directorio que prefieras, ejecuta el siguiente comando:\
   ```git clone https://github.com/Ugulberto/est-inmersion.git```\
3. Instala el lenguaje de programación python: https://www.python.org/downloads/\
4. Una vez instalado python, ejecuta el siguiente comando:\
   ```pip install --upgrade pip```\
   Y después este:\
  ```pip install numpy, tensorflow, scikit-learn, matplotlib```\
5. Entonces, ejecuta:
   ```python keras_nn_seno.py```\
y espera a obtener el resultado del entrenamiento.
