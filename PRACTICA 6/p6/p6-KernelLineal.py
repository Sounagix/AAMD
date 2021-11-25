import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D

# Cargamos los datos
dataset1 = loadmat ("ex6data1.mat")
X1 = dataset1['X']
Y1 = dataset1['y']

pos = np.array([X1[i] for i in range(len(X1))if Y1[i]==1])
neg = np.array([X1[i] for i in range(len(X1))if Y1[i]==0])

# Inicializacion del Kernel
kernel_lin = svm.SVC(C= 1, kernel = 'linear')
res = kernel_lin.fit(X1, Y1.flatten())

# Métodos de dibujado de Kernel
def RepresentData(pos,neg):
    plt.plot(pos[:, 0], pos [:, 1], "gx", label = "Positive Values")
    plt.plot(neg[:, 0], neg [:, 1], "ro", label = "Negative Values")
    plt.xlabel("X Values")
    plt.ylabel("Y Values")
    plt.legend()
    plt.grid(True)

def DrawLimit(svm, pos, neg, xMin, xMax, yMin, yMax):
    figure = plt.figure()
    X = np.linspace(xMin,xMax,200)
    Y = np.linspace(yMin,yMax,200)
    zVals = np.zeros(shape = (len(X), len(Y)))
    
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            aux = np.array([[X[i],Y[j]]])
            zVals[i][j] = float(svm.predict(aux))
    
    zVals = zVals.T
    RepresentData(pos,neg)
    a,b = np.meshgrid(X,Y)
    countour = plt.contour(X,Y,zVals)


DrawLimit(kernel_lin, pos, neg, -1,5,1,5)

kernel_lin = svm.SVC(C= 100, kernel = 'linear') 
kernel_lin.fit(X1, Y1.flatten())

DrawLimit(kernel_lin, pos, neg, -1,5,1,5)
plt.show()