import numpy as np
import  dill
import matplotlib.pyplot as plt
from Code.Q4.Problem4_a_b import dist_from_mean,distance

with open('kernel_4a.pkl', 'rb') as f:
    k = dill.loads(dill.load(f))

x = np.load('data.npy')
d=100
D = np.zeros((d,d))
for i in range(d):
    for j in range(d):
        D[i,j] = distance(x[i,:],x[j,:],k)

n = len(x)
d = len(x[0])
n_clust = 2
itr_n = 2
color=['r','b']
c_old = [list() for i in range(n_clust)]
np.random.shuffle(x)

#Initialization
for i in range(n_clust):
    c_old[i] = list(x[int(i*n/n_clust):int((i+1)*n/n_clust)])

c_new = [list() for i in range(n_clust)]
for itr in range(itr_n):
    for i in range(n):
        temp = np.asarray([dist_from_mean(x[i],c_old[j],k) for j in range(n_clust) if c_old[j]!=[]])
        c = np.argmin(temp)
        c_new[c].append(x[i])
    c_old = c_new
    c_plot = [np.asarray(j) for j in c_new]
    fig = plt.figure()
    for i in range(n_clust):
        plt.scatter(c_plot[i][:,0],c_plot[i][:,1],label=f'Cluster {i+1}',color=color[i])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Problem 4c')
    plt.legend()
    plt.savefig(f'Problem_4c_itr_{itr+1}.png')
