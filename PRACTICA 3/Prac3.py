from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import scipy . optimize as opt

#region ######################## 1.2 Clasificación de uno frente a todos ########################
def func_sigmoid(X):
    return 1.0/(1.0 + np.exp(-X))

def cost(theta, X, Y, reg):
    h = func_sigmoid(np.matmul(X, theta))
    m = X.shape[0]
    coste = (- 1 / m) * (np.dot(Y, np.log(h)) +
                               np.dot((1 - Y), np.log(1 - h))) + ((reg/2*m) * np.sum(np.power(theta[1:],2)))
    return coste

def gradient(theta, X, Y, reg):
    m = X.shape[0]
    h = func_sigmoid(np.matmul(X, theta))
    Taux = theta
    Taux[0] = 0
    aux = (1/m) * np.matmul(X.T, (np.ravel(h) - Y))
    return aux + ((reg/m) * Taux)

def getLabel(Y, etiqueta):
    y_etiqueta = np.ravel(Y)== etiqueta
    y_etiqueta = y_etiqueta *1
    return y_etiqueta

def evaluate(theta, X, Y):
    xs = func_sigmoid(np.matmul(X, theta))
    xspositivas = np.where(xs >= 0.5)
    xsnegativas = np.where(xs < 0.5)
    xspositivasexample = np.where (Y == 1)
    xsnegativasexample = np.where (Y == 0)

    porcentajePositivas = np.intersect1d(xspositivas, xspositivasexample).shape[0]/xs.shape[0]
    porcentajeNegativas = np.intersect1d(xsnegativas, xsnegativasexample).shape[0]/xs.shape[0]
    
    return porcentajeNegativas + porcentajePositivas

def oneVsAll(X, y, n_labels, reg):
    """
    oneVsAll entrena varios clasificadores por regresión logística con término
    de regularización 'reg' y devuelve el resultado en una matriz, donde
    la fila i-ésima corresponde al clasificador de la etiqueta i-ésima
    """
    m = X.shape[1]
    theta = np.zeros((n_labels, m))
    yLabels = np.zeros((y.shape[0], n_labels))

    # recorremos el numero de etiquetas y vemos si yj ∈ 0, 1 indica si el ejemplo de entrenamiento j-ésimo 
    # pertenece a la clase k (yj = 1) o a otra clase (yj = 0)
    # Al final cambiamos la etiqueta del 0 que corresponde a 10
    for i in range(n_labels):
        yLabels[:,i] = getLabel(y, i)
    yLabels[:,0] = getLabel(y, 10)

    # ahora lo que hacemos es coger la theta optima para los valores anteriores pero con el valor de regularizacion
    # tal y como haciamos en la regresion logistica
    for i in range(n_labels):
        theta_opt = opt.fmin_tnc(func=cost, x0=theta[i,:], fprime=gradient, args=(X, yLabels[:,i], reg))
        theta[i, :] = theta_opt[0]

    # a continuacion evaluamos los resultados
    evaluation = np.zeros(n_labels)
    for i in range(n_labels):
        evaluation[i] = evaluate(theta[i,:], X, yLabels[:,i])

    print("Evaluacion: ", evaluation)
    print("Evaluacion media: ", evaluation.mean())
    return 0

#endregion

#region ######################## 2. Redes neuronales ########################
def h(X, thetas1, thetas2):
    # primera capa de la red
    a1 = X
    # segunda capa de la red
    z2 = np.matmul(thetas1, np.insert(a1,0,1))
    a2 = func_sigmoid(z2)
    # tercera capa de la red
    z3 = np.matmul(thetas2, np.insert(a2,0,1))
    a3 = func_sigmoid(z3)
    return a3
#endregion

#region ######################## 1.1. Visualización de los datos ########################

# El fichero se carga con la función scipy.io.loadmat que devuelve un diccionario del que podemos extraer las matrices X y
data = loadmat('ex3data1.mat')

# X e Y son matrices. Cada matriz de 20×20 se ha desplegado para formar un vector de 400 componentes que ocupa una
# fila de la matriz X. X es una matriz de 5000×400 donde cada fila representa la imagen de un número escrito a mano.
# almacena los datos leídos en X, y
y = data['y']
X = data['X']

num_labels = 10
regularization = 0.1
oneVsAll(X, y, num_labels, regularization)

# se pueden consultar las claves del diccionario con data.keys()
print(data.keys())
# Selecciona aleatoriamente 10 ejemplos y los pinta
sample = np.random.choice(X.shape[0], 10)
plt.imshow(X[sample, :].reshape(-1, 20).T)
plt.axis('off')
#endregion

#region ######################## 2. Redes neuronales ########################
thetas = loadmat("ex3weights.mat")
print(thetas.keys())
thetas1 = thetas["Theta1"] # Theta1 es de dimensión 25 x 401
thetas2 = thetas["Theta2"] # Theta2 es de dimensión 10 x 26

aux = np.zeros(num_labels)

for i in range(num_labels):
    aux[i] = np.argmax(h(X[sample[i],:],thetas1,thetas2))

print("My guess are: ", (aux)+1)
numAciertos = 0

for i in range(X.shape[0]):
    aux2 = np.argmax(h(X[i,:],thetas1, thetas2))
    if(aux2+1) == y[i]:
        numAciertos +=1

print("Porcentaje de aciertos: ", numAciertos / X.shape[0])
#endregion

plt.show()