import numpy as np

class Xarxaneuronal2:
    def __init__(self):
        # Estructura neuronal
        self.n_entradas = 2  # neuronas entrada
        self.n_internas = 3  # neuronas de dentro esquema
        self.n_salidas = 2   # neuronas salida

        # Inicializo pesos y bias con valor aleatorio
        self.pesos_1 = np.random.rand(self.n_entradas, self.n_internas)
        # Los pesos me ayudan a determinar cuanto cada entrada contribuye a la salida.
        # en este caso representan los pesos entre la capa de entrada y las capa interior, y entre la capa interior y la capa salida.
        self.bias_1 = np.random.rand(self.n_internas)
        # Los bias me ayudan a que la red neuronal aprenda y se ajuste a diferentes datos, con esto evito que la red solo se limite a aprender patrones que pasen por el origen (0,0)
        self.pesos_2 = np.random.rand(self.n_internas, self.n_salidas)
        self.bias_2 = np.random.rand(self.n_salidas)

        # Tasa de aprendizaje
        self.alpha = 0.25 # Controla cuanto cambiaran los pesos y los bias durante el entrenamiento

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivada_sigmoid(self, x):  # Actualizar el gradiente durante la repro, basicamente sirve para calcular como deben ajustarse los pesos en funcion del error.
        return x * (1 - x)

    def forward_propagation(self, entradas):  # Calcula la salida de la red neuronal dada una entrada
        # Capa interior
        z1 = np.dot(entradas, self.pesos_1) + self.bias_1 # Hago servir el "np.dot()" para la multi de matrices, que me ayuda a realizar la funcion de forward_propagation.
        a1 = self.sigmoid(z1)

        # Capa salida
        z2 = np.dot(a1, self.pesos_2) + self.bias_2
        a2 = self.sigmoid(z2)

        return a1, a2

    def backward_propagation(self, entradas, salidas_deseadas):  #Ajusta los biases y los pesos segun el error de salida
        a1, a2 = self.forward_propagation(entradas)

        # Error en salida
        error_salida = salidas_deseadas - a2  # Aqui calculo la dferencia entre la salida deseada, mencionada abajo del codigo, y la salidas obtenidas. Es asi como consigo el error que quiero minimizar.
        delta_salida = error_salida * self.derivada_sigmoid(a2)

        # Error capa interior
        error_interior = np.dot(delta_salida, self.pesos_2.T) # Utilizo ".T" para hacer la traspuesta de la matriz y asi poder calcular el error de la capa interior.
        delta_interior = error_interior * self.derivada_sigmoid(a1)

        # Actualizo pesos y bias
        self.pesos_2 += self.alpha * np.dot(a1.T, delta_salida) # Aqui utilizo ".T" para actualizar los valores de los pesos y los bias en funcion de los gradientes de error.
        self.bias_2 += self.alpha * np.sum(delta_salida, axis=0) # Aqui acumulo el error de cada neurona de salida en funcion de la entrada. Igualar axis a 0 me permite que la suma se realice a lo largo de las filas.
        self.pesos_1 += self.alpha * np.dot(entradas.T, delta_interior)
        self.bias_1 += self.alpha * np.sum(delta_interior, axis=0)

    def entrenar(self, entradas, salidas_deseadas):
        for _ in range(10000):  # Aumento el número de iteraciones para un mejor entrenamiento
            self.backward_propagation(entradas, salidas_deseadas)

# Creo instancia
xarxa = Xarxaneuronal2()

# Entradas y salidas que quiero para XOR y AND
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
salidas_deseadas = np.array([[0, 0], [1, 0], [1, 0], [0, 1]])  # Salidas deseadas para XOR y AND

# Entreno la red neuronal
xarxa.entrenar(entradas, salidas_deseadas)

# Test xarxa con todas las combinaciones de entrada
for entrada in entradas:
    salidas = xarxa.forward_propagation(entrada.reshape(1, -1))[1]
    print(f"Entrada: {entrada}, Salida: {salidas}")