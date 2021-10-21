import matplotlib.pyplot as plt
import numpy as np
from pandas.io.parsers import read_csv
import scipy . optimize as opt
from sklearn.preprocessing import PolynomialFeatures 


# Lee un archivo csv pasando el nombre del fichero a leer y devuelve un array de numpy 
def leeCSV(file_name):
    """carga el fichero csv especificado y lo
     devuelve en un array de numpy"""
    valores = read_csv(file_name, header = None).to_numpy()
    return valores.astype(float)

#region METODOS####################### 1.2 FUNCION SIGMOIDE ########################################
def func_sigmoide(X):
    return 1 / (1 + np.exp(-X))

#endregion

#region METODOS####################### 1.3 CALCULO DE LA FUNCION DE COSTE Y SU GRADIENTE #####################
def cost(theta, X, Y):
    # H = sigmoid(np.matmul(X, np.transpose(theta)))
    H = func_sigmoide(np.matmul(X, theta))
    # cost = (- 1 / (len(X))) * np.sum( Y * np.log(H) + (1 - Y) * np.log(1 - H))
    cost = (- 1 / (len(X))) * (np.dot(Y, np.log(H)) +
                               np.dot((1 - Y), np.log(1 - H)))
    return cost

def gradient(theta, XX, Y):
    H = func_sigmoide(np.matmul(XX, theta))
    grad = (1 / len(Y)) * np.matmul(np.transpose(XX), H - Y)
    return grad
#endregion

#region METODOS####################### 1.4 CALCULO DEL VALOR OPTIMO Y PINTAR FRONTERA ########################
def pinta_frontera_recta(X, Y, theta):
 plt.figure()

 x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
 x2_min, x2_max = X[:, 2].min(), X[:, 2].max()

 xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
 np.linspace(x2_min, x2_max))

 h = func_sigmoide(np.c_[np.ones((  xx1.ravel().shape[0], 1)),
                                    xx1.ravel(),
                                    xx2.ravel()].dot(theta))
 h = h.reshape(xx1.shape)

 # el cuarto parámetro es el valor de z cuya frontera se
 # quiere pintar
 plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')

#endregion

#region METODOS####################### 1.5 EVALUACION DE LA REGRESION LOGISTICA ##############################
def percentage(theta):
    sigmoide = func_sigmoide(theta)
    return len(np.where(sigmoide >= 0.5))/len(sigmoide)



#endregion ###############################################################################################

#region METODOS####################### 2.2 Cálculo de la función de coste y su gradiente ############
def regularized_Grad(Theta, X, Y, lamb):
    g = gradient(Theta, X, Y)
    # versión regularizada de la regresión logística.
    r = (lamb/np.shape(X)[0]) * Theta
    g = g + r
    return g

def regularized_cost(Theta, X, Y, lamb):
    m = np.shape(X)[0]
    sR = cost(Theta, X, Y)
    reg = np.sum(Theta[1:] ** 2)
    r = (lamb * reg) / (2 * m)
    return sR + r

#endregion

#region METODOS###################### 2.4. Efectos de la regularización #############################
def pinta_frontera_circulo(X, Y, Theta, poly):
    plt.figure()
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = func_sigmoide(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(Theta))                   
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')

#endregion

#region ####################### 1.1 VISUALIZACION DE LOS VALORES ####################################

# valores = leeCSV("ex2data1.csv")
# X = valores[:, :2] # X es una matriz de [100, 2] (resultado de los examenes 1 y 2)
# Y = valores[:, 2] # Y es una matriz/vector [100, 1] (admitido¿?)
# #print(valores.shape) # 100 , 3 (son los datos de 100 alumnos, en 2 examenes, columna 0 y 1, y si fueron admitidos, columna 2)

# #Añadir fila de unos a la matriz X tal que la columna 0 es todo 1's
# X = np.hstack([np.ones([np.shape(X)[0], 1]), X]) # [100, 3]
# n = np.shape(X)[1]
# Theta = np.zeros(n) # [3]

# #print(cost(Theta, X, Y)
# #print(gradient(Theta, X, Y))


# #Obtiene un vector con los índices de los ejemplos positivos
# pos = np.where(Y == 1)
# #Obtiene un vector con los índices de los ejemplos negativos
# neg = np.where(Y == 0)

# # obtener el valor de los parámetros Theta que minimizan la función de coste para la regresión logística
# result = opt.fmin_tnc(func = cost, x0 = Theta, fprime = gradient, args = (X, Y))
# theta_opt = result[0]
# print(percentage(theta_opt))

# pinta_frontera_recta(X, Y, theta_opt)

# # Dibuja los ejemplos positivos
# plt.scatter(X[pos , 1] , X[pos , 2] , marker='+' , c='k', label="Admitted")
# # Dibuja los ejemplos negativos
# plt.scatter(X[neg , 1] , X[neg , 2] , marker='o' , c='g', label="No Admitted")
# #plt.legend()
# plt.show()
# plt.savefig("frontera.png")
# plt.close()
#endregion

#region ####################### 2.0 Reg Log regularizada #####################################
valores2 = leeCSV("ex2data2.csv")
X = valores2[:, :2] # X es una matriz de [100, 2] (resultado de los examenes 1 y 2)
Y = valores2[:, 2] # Y es una matriz/vector [100, 1] (admitido¿?)
# print(valores2.shape) # 100 , 3 (son los datos de 100 alumnos, en 2 examenes, columna 0 y 1, y si fueron admitidos, columna 2)
#endregion

#region ####################### 2.1 Mapeo de los atributos #####################################
# Añade los ejemplos hasta la sexta potencia
poly = PolynomialFeatures(6)
polyFit  = poly.fit_transform(X)
m = np.shape(polyFit)[0]
n = np.shape(polyFit)[1]
#endregion

#region ####################### 2.2. Cálculo de la función de coste y su gradiente #############
lamb = 10
Theta = np.zeros(n)

coste_r = regularized_cost(Theta,polyFit, Y, lamb)
gradiente_r = regularized_Grad(Theta, polyFit ,Y, lamb)
print("Coste regularizado: {}".format(coste_r))
print("Gradiente regularizado: {}".format(gradiente_r))
#endregion

#region ####################### 2.3. Cálculo del valor óptimo de los parámetros #################
result = opt.fmin_tnc(func=regularized_cost,x0=Theta, fprime=regularized_Grad, args=(polyFit,Y, lamb))
thetaOpt = result[0]
print("Theta optimo: {}".format(thetaOpt))
#endregion

#region ####################### 2.4. Efectos de la regularización ##############################
pinta_frontera_circulo(X ,Y ,thetaOpt ,poly )
#Obtiene un vector con los índices de los ejemplos positivos
pos = np.where(Y == 1)
#Obtiene un vector con los índices de los ejemplos negativos
neg = np.where(Y == 0)
plt.scatter(X[pos , 0] , X[pos , 1] , marker='+' , c='k', label="Y = 1")
# Dibuja los ejemplos negativos
plt.scatter(X[neg , 0] , X[neg , 1] , marker='o' , c='g', label="Y = 0")
plt.legend()
plt.show()
plt.close()
#endregion

