import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D

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

dataset2 = loadmat ("ex6data2.mat")
X2 = dataset2['X']
Y2 = dataset2['y']

pos = np.array([X2[i] for i in range(len(X2))if Y2[i]==1])
neg = np.array([X2[i] for i in range(len(X2))if Y2[i]==0])

sigma = 0.1
gaussiananKernel = svm.SVC(C = 1, kernel = 'rbf', gamma = 1/(2*sigma **2))
gaussiananKernel.fit(X2,Y2.flatten())

DrawLimit(gaussiananKernel, pos, neg, 0.0,1.0,0.4,1.0)
plt.show()