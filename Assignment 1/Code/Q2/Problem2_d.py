import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from matplotlib import cm

def get_data(n,view_plt=False):
    np.random.seed(0)
    w_mean = np.zeros(2)
    cov = np.eye(2)
    w = np.random.multivariate_normal(w_mean,cov)
    b = np.random.normal(0,1)

    x_p1 = []
    x_p2 = []
    x_n1 = []
    x_n2 = []
    y = []
    x = []
    for i in range(n):
        x_temp=np.random.uniform(low=-3,high=3,size=2)
        x.append(x_temp)
        if (((x_temp[0]**2)+(0.5*(x_temp[1]**2)))<=2):
            y.append(1)
        else:
            y.append(-1)
        if y[i]>0:
            x_p1.append(x_temp[0])
            x_p2.append(x_temp[1])
        else:
            x_n1.append(x_temp[0])
            x_n2.append(x_temp[1])
    if view_plt:
        plt.scatter(x_p1,x_p2,color='red',label=1)
        plt.scatter(x_n1,x_n2,color='blue',label = -1)

        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Problem 2_c')
        plt.legend()
        plt.show()

    x = np.asarray(x).T.pow(2)
    y = np.asarray(y)
    return x,y


n = 100
d = 2
c = 5

#Get train data
x,y = get_data(n,True)

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
x_test,y_test = get_data(n_test)
true = 0
for i in range(n_test):
  y_pred = np.sign(np.dot(w.T,x_test[:,i])+b)
  if y_pred == y_test[i]:
    true+=1

print(f'Accuracy on test set : {true/n_test}')