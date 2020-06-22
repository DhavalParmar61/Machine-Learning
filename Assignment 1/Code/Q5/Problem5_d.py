import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot

def vect_k(x,d):
    k = np.power(1+x*d,2)
    return k

def matrix_K(x):
    n = len(x)
    K = np.zeros((n,n))
    for i in range(n):
        k = vect_k(x,x[i])
        K[i,:]=k
    return K

f = open('data_c1.txt','r')
c1 =[]
for line in f:
    line = line.replace('\n','')
    c1.append(float(line))
c1 = np.asarray(c1)
f.close()

f = open('data_c2.txt','r')
c2 =[]
for line in f:
    line = line.replace('\n','')
    c2.append(float(line))
c2 = np.asarray(c2)
f.close()

x = np.hstack([c1,c2])
lp = len(c1)
ln = len(c2)
l = lp+ln
y = np.zeros(l)
y[0:lp]=1
y[lp:l]=-1
K = matrix_K(x)
# For D
D = np.zeros((l,l))
t = np.zeros(l)
for i in range(l):
    if y[i]==1:
        D[i,i] = 2*ln/l
        t[i] = 1/lp
    else:
        D[i,i] = 2*lp/l
        t[i] = 1/ln

# fro CP
cp = np.zeros((l,l))
cn = np.zeros((l,l))
for i in range(i):
    for j in range(l):
        if y[i]==1 and y[j]==1:
            cp[i,j] = 2*ln/(lp*l)
        if y[i]==-1 and y[j]==-1:
            cn[i,j] = 2*lp/(ln*l)

B = D-cp-cn
reg = 0.01

alpha = np.dot(np.linalg.inv(np.dot(B,K)+(reg*np.eye(l))),y)
b = 0.5*multi_dot([alpha,K,t])
y_pred = []
for i in range(l):
    k_vect = vect_k(x,x[i])
    y_pred.append((multi_dot([k_vect,alpha])-b))

y_pred_p =[]
y_pred_n =[]
for i in range(l):
    if y[i]==1:
        y_pred_p.append(y_pred[i])
    else:
        y_pred_n.append(y_pred[i])
y_pred_p = np.asarray(y_pred_p)
y_pred_n = np.asarray(y_pred_n)

thresold = (np.mean(y_pred_p)/np.std(y_pred_p)+np.mean(y_pred_n)/np.std(y_pred_n))/2

true = 0
for i in range(len(y)):
    if y_pred[i]>=thresold and y[i]==1:
        true+=1
    elif y_pred[i]<thresold and y[i]==-1:
        true+=1

print(f'Classification Accuracy:{true*100/l}%')

# Ploting 1d representation of data
fig = plt.figure()
plt.scatter(y_pred_p,np.zeros(len(y_pred_p)),label ='c1',color='r')
plt.scatter(y_pred_n,np.zeros(len(y_pred_n)),label ='c2',color='b')
plt.scatter(thresold,0,label='Thresold',color='g')
plt.xlabel('x')
plt.title('Problem 5d Representation of Data')
plt.legend()
plt.savefig('Problem5d_repr_data.png')

#Ploting alpha
fig = plt.figure()
plt.scatter(x,alpha,label='Alpha')
plt.xlabel('x')
plt.ylabel('Alpha Value')
plt.title('Problem5d Alpha Value')
plt.legend()
plt.savefig('Problem5d_alpha.png')