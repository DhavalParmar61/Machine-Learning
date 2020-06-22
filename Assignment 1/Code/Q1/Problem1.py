import  dill
import numpy as np
from numpy import linalg
from numpy.linalg import multi_dot

n = 100
iter_n=10
d = 3
count = 0

with open('./function5.pkl', 'rb') as f:
    k = dill.loads(dill.load(f))

samp_file = open('./k5sampler.pkl','rb')
k5_sampler = dill.loads(dill.load(samp_file))

for iter in range(iter_n):
    A = np.zeros((n,n))
    kernel = True

    #For function 5
    '''
    x = np.zeros((d,n))
    for i in range(n):
        x[:,i] = np.squeeze(k5_sampler())
    '''

    #For function 1 to 4
    for i in range(n):
        x = np.random.uniform(low=-5,high=5,size=(d,n))

    for i in range(n):
        for j in range(n):
            x1 = x[:,i].reshape(-1,1)
            x2 = x[:,j].reshape(-1,1)
            A[i,j] = k(x1,x2)
    for i in range(n):
        for j in range(n):
            if A[i,j]!=A[j,i]:
                kernel=False
                break
        if not kernel:
            break

    if kernel:
        for i in range(n):
            if A[i,i]<0:
                kernel = False

    if kernel:
        eig_value, eig_vect = linalg.eig(A)
        eig_value = np.real(eig_value)
        eig_value = [0 if 0>i>-10**-6 else i for i in eig_value]
        for i in range(len(eig_value)):
            if(eig_value[i]<0):
                kernel =False
    if kernel == True:
        count = count+1

if count>(iter_n/2):
    print('It is a kernel.')
else:
    print('It is not a kernel.')