import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot

def vect_k(x,d):
    #k = np.power(x-d,4)
    c = 0.065
    k = 1 - (np.power((x-d),2)/(np.power((x-d),2)+c))
    return k

def matrix_K(x):
    n = len(x)
    K = np.zeros((n,n))
    for i in range(n):
        k = vect_k(x,x[i])
        K[i,:]=k
    return K

f = open('train_data.txt','r')
x= []
y = []
for line in f :
    x_temp,y_temp = line.split(' ')
    x.append(x_temp)
    y.append(y_temp)
x = np.asarray(x).astype(np.float)
y = np.asarray(y).astype(np.float)
f.close()

n = len(x)
l = 10**-10
K_mat = matrix_K(x)
a = np.dot(np.linalg.inv(K_mat+(l*np.eye(n))),y)
y_pred = []
for i in range(n):
    k_vec = vect_k(x,x[i])
    y_temp = np.dot(k_vec,a)
    y_pred.append(y_temp)

y_pred = np.asarray(y_pred)
#MSE on train
mse = 0.5*multi_dot([a,K_mat,K_mat,a])-multi_dot([a,K_mat,y])+0.5*np.dot(y,y)+0.5*l*multi_dot([a,K_mat,a])
mse = mse/n
print(f'MSE for train set : {mse}')

# For Test data
f = open('test_data.txt','r')
x_test = []
y_test = []
for line in f :
    x_temp,y_temp = line.split(' ')
    x_test.append(x_temp)
    y_test.append(y_temp)
x_test = np.asarray(x_test).astype(np.float)
y_test = np.asarray(y_test).astype(np.float)
f.close()

y_test_pred = []
for i in range(n):
    k_vec = vect_k(x,x_test[i])
    y_temp = np.dot(k_vec,a)
    y_test_pred.append(y_temp)

y_test_pred = np.asarray(y_test_pred)
#MSE on train
mse = 0.5*multi_dot([a,K_mat,K_mat,a])-multi_dot([a,K_mat,y_test])+0.5*np.dot(y_test,y_test)+0.5*l*multi_dot([a,K_mat,a])
mse = mse/n
print(f'MSE for test set : {mse}')

# For Plot
x_plot = np.linspace(-1,1,100)
n = len(x)
y_plot = []
for i in range(n):
    k_vec = vect_k(x,x_plot[i])
    y_temp = np.dot(k_vec,a)
    y_plot.append(y_temp)

y_plot = np.asarray(y_plot)
fig = plt.figure()
plt.plot(x_plot, y_plot, label='f(x)', color='r')
plt.scatter(x, y, label='Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Problem 3d')
plt.legend()
plt.savefig(f'Problem3d.png')
