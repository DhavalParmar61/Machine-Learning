import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from matplotlib import cm
import matplotlib.pyplot as plt

n = 100
d = 2
c = 5

#Get train data
x = []
y = []
f = open('2a_train_data.txt','r')
for line in f:
    x1,x2,y_temp=line.split(' ')
    x.append(np.asarray([x1,x2],dtype=float))
    y.append(y_temp)
x = np.asarray(x).T
y = np.asarray(y,dtype=float)

P = np.zeros((n+d+1,n+d+1))
P[0:2,0:2] = 0.5*np.eye(2)

q = np.zeros(n+d+1)
q[d:n+d] = c

G = np.zeros((2*n,n+d+1))
for i in range(n):
    G[i,n+d] = -y[i]
    G[i,0] = -x[0,i]*y[i]
    G[i,1] = -x[1,i]*y[i]
G[0:n,d:n+d] = -1*np.eye(n)
G[n:2*n,d:n+d] = -1*np.eye(n)

h = np.zeros(2*n)
h[0:n] = -1

P = matrix(P,tc='d')
q = matrix(q,tc='d')
G = matrix(G,tc='d')
h = matrix(h,tc='d')

sol = solvers.qp(P,q,G,h)

var = sol['x']
w = var[0:d]
zeta = var[d:n+d]
b = var[n+d-1]

print(f'w : {w}')
print(f'b : {b}')

f = lambda x,y: w[0]*x+w[1]*y+b
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1,2,1,projection='3d')
x_value = np.linspace(-3,3,100)
y_value = np.linspace(-3,3,100)
x_grid,y_grid = np.meshgrid(x_value,y_value)
z_values = f(x_grid,y_grid)
surf = ax.plot_surface(x_grid, y_grid, z_values,rstride=5, cstride=5,linewidth=0, cmap=cm.plasma)
ax = fig.add_subplot(1,2,2)
plt.contourf(x_grid, y_grid, z_values, 30,cmap=cm.plasma)
fig.colorbar(surf, aspect=18)
plt.tight_layout()


# Results on Train data
true = 0
for i in range(n):
  y_pred = np.sign(np.dot(w.T,x[:,i])+b)
  if y_pred == y[i]:
    true+=1

print(f'Accuracy on train set : {true/n}')

# Results on Test data
n_test = 50
x_test = []
y_test = []
f = open('2a_test_data.txt','r')
for line in f:
    x1,x2,y_temp=line.split(' ')
    x_test.append(np.asarray([x1,x2],dtype=float))
    y_test.append(y_temp)
x_test = np.asarray(x_test).T
y_test = np.asarray(y_test,dtype=float)

true = 0
for i in range(n_test):
  y_pred = np.sign(np.dot(w.T,x_test[:,i])+b)
  if y_pred == y_test[i]:
    true+=1

print(f'Accuracy on test set : {true/n_test}')