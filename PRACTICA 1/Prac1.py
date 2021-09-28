import matplotlib.pyplot as plt
import numpy as np
#   Para importar el parseador de frame a numpy
from pandas.io.parsers import read_csv
#   
from mpl_toolkits.mplot3d import Axes3D
#
from matplotlib.ticker import LinearLocator,FormatStrFormatter

from matplotlib import cm

# Lee un archivo csv pasando el nombre del fichero a leer y devuelve un array de numpy 
def leeCSV(file_name):
    """carga el fichero csv especificado y lo
     devuelve en un array de numpy"""
    valores = read_csv(file_name, header = None).to_numpy()
    return valores.astype(float)

def regLinealUnaVariable():
    datos = leeCSV("ex1data1.csv")
    #   los arrays de la siguiente manera [:, 0] nos devuelve en X los elementos de la primera columna
    X = datos[:, 0]
    #   los arrays de la siguiente manera [:, 1] nos devuelve en Y los elementos de la primera columna
    Y = datos[:, 1]
    #   m => es el rango de puntos que existen
    m = len(X)
    #   Ratio de aprendizaje del algoritmo de descenso de gradiente
    alpha = 0.01
    #   theta_0 => eje x
    #   theta_1 => eje z
    theta_0 = theta_1 = 0
    #   
    for _ in range(1500):
        sum_0 = sum_1 = 0
        #   h0(x) por el modelo lineal es = theta_0 + theta_1 * x
        for i in range(m):
            #   valores de los sumatorios para la formula del descenso de gradiente
            sum_0 += (theta_0 + theta_1 * X[i]) - Y[i]
            sum_1 += ((theta_0 + theta_1 * X[i]) - Y[i]) * X[i]
        
        #   formulas del gradiente para theta0 y theta1
        theta_0 = theta_0 - (alpha / m) * sum_0
        theta_1 = theta_1 - (alpha / m) * sum_1
    
    #   dibujamos la grafica
    plt.plot(X, Y, "x")
    min_x = min(X)
    max_x = max(X)
    min_y = theta_0 + theta_1 * min_x
    max_y = theta_0 + theta_1 * max_x
    plt.plot([min_x, max_x], [min_y, max_y])
    plt.savefig("resultado.png")
    
    #plt.contourf(makeData([-10, 10], [-1, 4], [min_x, max_x], [min_y, max_y]), np.logspace(-2, 3, 20))
    #plt.savefig("Countour.png")

#   Para determinar el coste de un trazo en la gradiente
def coste(X, Y, Theta):
    H = Theta[0] + X * Theta[1]
    Aux = (H - Y) ** 2
    return Aux.sum() / (2 * len(X))

#   Introduciomos los datos en la gráfica y mostramos el resultado
def countourGraph(a, b, c):
    plt.contour(a, b, c, np.logspace(-2, 3, 20), colors='blue')
    plt.savefig("representacion2D.png")

def surfaceGraph(a, b, c):
    fig = plt.figure()
    ax = fig.gca(projection='3d') 
    surf = ax.plot_surface(a, b, c, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
    # Customizar el eje Z
    ax.set_zlim(0, 700)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig("representacion3D.png")

#   Genera las matrices Theta0, Theta1, conste para generar un plot en 3d
#   del coste para valores de theta_0 en el intervalo t0_range y valores de theta_1
#   en el intervalor de t1_range
def makeData(t0_range, t1_range, X ,Y):
    step = 0.1
    # np.arange nos crea un numpy array desde t0_range[0] hasta t0_range[1] de step en step
    # ejemplo: np.arange(2, 10, 3) da de resultado [2, 5, 8]
    Theta0 = np.arange(t0_range[0], t0_range[1], step) # [-10, -9.9, -9.8....10]
    Theta1 = np.arange(t1_range[0], t1_range[1], step) # [-1, -0.9, -0.8...4]
    #   Para generar el grid de valores de theta0 = 0 y theta1 = 0 en sus intervalos
    Theta0, Theta1 = np.meshgrid(Theta0, Theta1)

    #   Creamos una copia de theta0
    Coste = np.empty_like(Theta0)
    #   Para todos los elementos de theta0
    for ix, iy in np.ndindex(Theta0.shape):
        #   Actualizamos valor de coste
        Coste[ix, iy] = coste(X, Y ,[Theta0[ix, iy], Theta1[ix, iy]])
    
    #   Devolvemos la tupla
    return [Theta0, Theta1, Coste]

#regLinealUnaVariable()

valores = leeCSV("ex1data1.csv")
t0_range = [-10, 10]
t1_range = [-1, 4]
X = valores[:, 0]
Y = valores[:, 1]

data = makeData(t0_range, t1_range, X, Y)

# Dibujamos la grafica en 2D
countourGraph(data[0], data[1], data[2])
# Dibujamos la grafica en 3D
surfaceGraph(data[0], data[1], data[2])

plt.show()