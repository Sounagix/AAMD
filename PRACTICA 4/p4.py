import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as opt
import checkNNGradients
import os

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pandas.io.parsers import read_csv
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from scipy.optimize import minimize

def func_sigmoid(X):
    return 1.0/(1.0 + np.exp(-X))

# método que aplica el algoritmo de propagacion hacia delante en una red neuronal
# ForwardProp es una funcion de hipotesis que utiliza un valor de entrada para predecir 
# una salida mediante una matriz de pesos. Tambien añadira un termino de sesgo.
def forward_propagate(Theta1, Theta2, X):
    z1 = Theta1.dot(X.T)
    a1 = func_sigmoid(z1)
    tuple = (np.ones(len(a1[0])), a1)
    a1 = np.vstack(tuple)
    z2 = Theta2.dot(a1)
    a2 = func_sigmoid(z2)
    return z1, a1, z2, a2

#cálculo del coste en una red neuronal
def NeuralNetworkCost(params, n_entries, n_hidden, n_et, X, Y, reg):
    theta1, theta2 = unroll_thetas(params, n_entries, n_hidden, n_et)

    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    j = 0
    for i in range(len(X)):
        j += (-1 / (len(X))) * (np.dot(Y[i], np.log(h[i])) + np.dot((1 - Y[i]), np.log(1 - h[i])))

    j += (reg / (2*len(X))) + ((np.sum(np.square(theta1[:,1:]))) + (np.sum(np.square(theta2[:,1:]))))

    return j


# main
# numero de etiquetas
num_labels = 10

data = loadmat('ex4data1.mat')
y = data['y'].ravel() # el metodo ravel hace que y pase de un shape(5000,1) a (5000,)
X = data['X']
thetas = loadmat("ex4weights.mat")
thetas1 = thetas["Theta1"] # Theta1 es de dimensión 25 x 401
thetas2 = thetas["Theta2"] # Theta2 es de dimensión 10 x 26

m = X.shape[0]
y = (y - 1)
y_onehot = np.zeros((m, num_labels))  # 5000 x 10

for i in range(m):
    y_onehot[i][y[i]] = 1


Thetas = [thetas1, thetas2]

unrolled_Thetas = [Thetas[i].ravel() for i,_ in enumerate(Thetas)]