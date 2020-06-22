import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt import solvers
from matplotlib import cm

def k(x,y):
  d = 2
  a = 1
  b = 1
  k = np.power((a+b*np.dot(x,y)),d)
  return k


n = 100
d = 2
c = 3

#Get train data
x = []
y = []
f = open('2c_train_data.txt','r')
for line in f:
    x1,x2,y_temp=line.split(' ')
    x.append(np.asarray([x1,x2],dtype=float))
    y.append(y_temp)
x = np.asarray(x).T
y = np.asarray(y,dtype=float)

P = np.zeros((n,n))
for i in range(n):
  for j in range(n):
    P[i,j]= 0.5*y[i]*y[j]*k(x[:,i],x[:,j])

q = -1*np.ones(n)

G = np.zeros((2*n,n))
G[0:n,:] = -1*np.eye(n)
G[n:2*n,:] = np.eye(n)

h = np.zeros(2*n)
h[n:2*n] = c

A = np.asarray([y[i] for i in range(n)]).reshape(-1,n)
b = 0

P = matrix(P,tc='d')
q = matrix(q,tc='d')
G = matrix(G,tc='d')
h = matrix(h,tc='d')
A = matrix(A,tc='d')
b = matrix(b,tc='d')

sol = solvers.qp(P,q,G,h,A,b)
alpha = sol['x']

for j in range(n):
  if alpha[j]<c and alpha[j]>0:
    b = y[j] - np.sum([alpha[i]*y[i]*k(x[:,j],x[:,i]) for i in range(n)])
    break
print(f'b = {b}')


# Results on Train data
true = 0
for i in range(n):
  y_pred = np.sign(np.sum([alpha[j]*y[j]*k(x[:,i],x[:,j]) for j in range(n)])+b)
  if y_pred == y[i]:
    true+=1

print(f'Accuracy on train set : {true/n}')

# Results on Test data
n_test = 50
x_test = []
y_test = []
f = open('2c_test_data.txt','r')
for line in f:
    x1,x2,y_temp=line.split(' ')
    x_test.append(np.asarray([x1,x2],dtype=float))
    y_test.append(y_temp)
x_test = np.asarray(x_test).T
y_test = np.asarray(y_test,dtype=float)

true = 0
for i in range(n_test):
  y_pred = np.sign(np.sum([alpha[j]*y[j]*k(x_test[:,i],x[:,j]) for j in range(n)])+b)
  if y_pred == y_test[i]:
    true+=1

print(f'Accuracy on test set : {true/n_test}')