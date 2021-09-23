import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv



# Lee un archivo csv pasando el nombre del fichero a leer y devuelve un array de numpy 
def readCSV(file_name):
    """carga el fichero csv especificado y lo
     devuelve en un array de numpy"""
    valores = read_csv(file_name, header = None).to_numpy()
    return valores.astype(float)

def descensoGradiente():
    datos = readCSV("ex1data1.csv")
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
        # h0(x) por el modelo lineal es = theta_0 + theta_1 * x
        for i in range(m):
            # valores de los sumatorios para la formula del descenso de gradiente
            sum_0 += (theta_0 + theta_1 * X[i]) - Y[i]
            sum_1 += ((theta_0 + theta_1 * X[i]) - Y[i]) * X[i]
        
        # formulas del gradiente para theta0 y theta1
        theta_0 = theta_0 - (alpha / m) * sum_0
        theta_1 = theta_1 - (alpha / m) * sum_1
    
    # dibujamos la grafica
    plt.plot(X, Y, "x")
    min_x = min(X)
    max_x = max(X)
    min_y = theta_0 + theta_1 * min_x
    max_y = theta_0 + theta_1 * max_x
    plt.plot([min_x, max_x], [min_y, max_y])
    plt.savefig("resultado.png")   
            


#   Genera las matrices Theta0, Theta1, conste para generar un plot en 3d
#   del coste para valores de theta_0 en el intervalo t0_range y valores de theta_1
#   en el intervalor de t1_range
def makeData(t0_range, t1_range,X,Y):
    step = 0.1


#arrayValores =  read_csv("ex1data1.csv")
#print(arrayValores)

descensoGradiente()