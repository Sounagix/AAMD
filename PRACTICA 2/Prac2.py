import matplotlib.pyplot as plt
import numpy as np
from numpy.core import numeric
#   Para importar el parseador de frame a numpy
from pandas.io.parsers import read_csv
#   
from mpl_toolkits.mplot3d import Axes3D
#
#from matplotlib.ticker import LinearLocator,FormatStrFormatter

#from matplotlib import cm

#import numpy.linalg as linalg

# Lee un archivo csv pasando el nombre del fichero a leer y devuelve un array de numpy 
def leeCSV(file_name):
    """carga el fichero csv especificado y lo
     devuelve en un array de numpy"""
    valores = read_csv(file_name, header = None).to_numpy()
    return valores.astype(float)

################################# 1.2 FUNCION SIGMOIDE ########################################
def func_sigmoide():

    return 1

################################################################################################

valores = leeCSV("ex2data1.csv")
X = valores[:, :2] #
Y = valores[:, 2]
print(valores.shape) # 100 , 3 (son los datos de 100 alumnos, en 2 examenes, columna 0 y 1, y si fueron admitidos, columna 2)


######################### 1.1 VISUALIZACION DE LOS VALORES ####################################

#Obtiene un vector con los índices de los ejemplos positivos
pos = np.where(Y == 1)
#Obtiene un vector con los índices de los ejemplos negativos
neg = np.where(Y == 0)

# Dibuja los ejemplos positivos
plt.scatter(X[pos , 0] , X[pos , 1] , marker='+' , c='k', label="Admitted")
# Dibuja los ejemplos negativos
plt.scatter(X[neg , 0] , X[neg , 1] , marker='o' , c='g', label="No Admitted")
plt.legend()
plt.show()

################################################################################################