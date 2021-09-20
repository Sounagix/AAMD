import numpy as np
import time
import random as rd
import matplotlib.pyplot as plt

def compara_tiempos():
    puntos = 8000
    puntos_totales = 10000
    sizes = np.linspace(100, 10000000, 20)
    times_buc = []
    times_vec = []
    for size in sizes:
        x1 = np.random.uniform(1, 100)
        x2 = np.random.uniform(1, 100)
        times_buc += [integra_mc_buc(funcion, x1, x2, puntos_totales)]
        times_vec += [integra_mc_vec(funcion, x1, x2, puntos_totales)]
        puntos_totales += puntos
    
    plt.figure()
    plt.scatter(sizes, times_buc, c='red', label='bucle')
    plt.scatter(sizes, times_vec, c='blue', label='vector')
    plt.legend()
    plt.savefig('time.png')

#funcion a pasar a la integral, funcion cuadrado
def funcion(x):
    return x*x

#funcion iterativa usando bucle
def integra_mc_buc(fun, a, b, num_puntos=10000):
    tic = time.process_time()
    maximo = fun(b)
    areaRectangulo = (b - a) * maximo
    nDebajo = 0
    for i in range(num_puntos):
        x = round(np.random.uniform(a, b))
        y = round(np.random.uniform(0, maximo))
        if y < fun(x):
            nDebajo = nDebajo + 1
    
    I = (nDebajo / num_puntos) * areaRectangulo
    print(I)
    toc = time.process_time()
    #tiempoEjecucion = 1000 * (toc - tic)
    #print('Tiempo de ejecucion en bucle: ' + str(tiempoEjecucion) + '\n' + 'Resultado integral: ' + str((nDebajo / num_puntos) * areaRectangulo))
    return 1000 * (toc - tic)  

#funcion que utiliza operaciones entre vectores
def integra_mc_vec(fun, a, b, num_puntos=10000):
    tic = time.process_time()
    maximo = fun(b)
    areaRectangulo = (b - a) * maximo
    arrX = np.random.uniform(a, b + 1, num_puntos)
    arrY = np.random.uniform(0, maximo + 1, num_puntos)
    nDebajo = np.sum(arrY < fun(arrX))
    I = (nDebajo / num_puntos) * areaRectangulo
    print(I)
    toc = time.process_time()
    #tiempoEjecucion = 1000 * (toc - tic)

    #print('Tiempo de ejecucion en vector: ' + str(tiempoEjecucion) + '\n' + 'Resultado integral: ' + str((nDebajo / num_puntos) * areaRectangulo))
    return 1000 * (toc - tic)

compara_tiempos()