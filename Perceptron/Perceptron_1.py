import numpy as np


class Xarxaneuronal1:
    def __init__(self):
        # Estructura neuronal
        self.n_entradas = 2
        self.n_ocultas = 3
        self.n_salidas = 2

        # Inicializar pesos y bias con valor 1
        self.pesos_1 = np.ones((self.n_entradas, self.n_ocultas))
        self.bias_1 = np.ones((self.n_ocultas))
        self.pesos_2 = np.ones((self.n_ocultas, self.n_salidas))
        self.bias_2 = np.ones((self.n_salidas))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, entradas):
        # Capa oculta
        z1 = np.dot(entradas, self.pesos_1) + self.bias_1
        a1 = self.sigmoid(z1)

        #Capa salida
        z2 = np.dot(a1, self.pesos_2) + self.bias_2
        a2 = self.sigmoid(z2)

        return a2
# Creamos instancia de clase
xarxa = Xarxaneuronal1()

entradas = np.array([0, 1])
salidas = xarxa.forward_propagation(entradas)
print(salidas)