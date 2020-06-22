import  dill
import numpy as np
import math

def distance(x,y,k):
    '''
    :param x: belongs to Rd
    :param y: belongs to Rd
    :param k: kernel function
    :return: distance between  φ(x) and φ(y)
    '''
    d = k(x,x)+k(y,y)-2*k(x,y)
    d = math.sqrt(d)
    return d

def dist_from_mean(x,y,k):
    '''
    :param x: belongs to Rd
    :param y: list of yi where yi belongs to Rd
    :return: distance between x and mean of y
    '''
    l = len(y)
    d = 0
    for i in range(l):
        d += distance(x,y[i],k)
    d = d/l
    return d

if __name__ == "__main__":
    d = 10
    E = np.eye(d)
    with open('kernel_4a.pkl','rb') as f:
        k = dill.loads(dill.load(f))

    D = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            D[i,j] = distance(E[:,i],E[:,j],k)

    print(f'Sum of all entries of D : {np.sum(D)}')

    E = list(E)
    d = 0
    for i in range(d):
        d += dist_from_mean(E[i],E,k)

    print(f'Sum of di : {d}')