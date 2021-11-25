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


dataset3 = loadmat ("ex6data3.mat")
X3 = dataset3['X']
Y3 = dataset3['y']
xTest = dataset3['Xval'] 
yTest = dataset3['yval'] 
pos = np.array([X3[i] for i in range(len(X3))if Y3[i]==1])
neg = np.array([X3[i] for i in range(len(X3))if Y3[i]==0])

cVals = np.array([0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30.])
sigmaVals = np.copy(cVals)

bestC = 0.01
bestSigma = 0.01
bestScore = -1
# Buscamos el mejor valor de c y de sigma para el SVM y así conseguir un kernel gaussiano de mayor precisión
for c in  cVals:
    for sigma in sigmaVals:
        gamma = 1/(2*sigma **2)
        auxKernel = svm.SVC(C = c, kernel = 'rbf', gamma = gamma)
        auxKernel.fit(X3, Y3.flatten())
        score = auxKernel.score(xTest,yTest)
        if (score > bestScore):
            bestC = c
            bestSigma = sigma   
            bestScore = score


gKernel = svm.SVC (C = bestC, kernel = 'rbf', gamma =  1/(2*bestSigma **2))
gKernel.fit(X3, Y3.flatten())

print(bestC)
DrawLimit(gKernel, pos, neg, -0.65, 0.3, -0.69, 0.61)
plt.show()