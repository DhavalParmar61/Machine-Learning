import numpy as np
import matplotlib.pyplot as plt

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


fig = plt.figure()
c1_p = np.vstack([c1,np.power(c1,2)]).T
c2_p = np.vstack([c2,np.power(c2,2)]).T
plt.scatter(c1_p[:,0],c1_p[:,1],label='C1',color='r')
plt.scatter(c2_p[:,0],c2_p[:,1],label='C2',color='b')
plt.title('Problem 5b')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.savefig('Problem_5b.png')

d=2
m1 = np.mean(c1_p,axis=0)
m2 = np.mean(c2_p,axis=0)
Sw = np.zeros((d,d))
for i in range(len(c1_p)):
    Sw += np.dot((c1_p[i]-m1)[:,None],(c1_p[i]-m1)[None,:])
for i in range(len(c2_p)):
    Sw += np.dot((c2_p[i]-m2)[:,None],(c2_p[i]-m2)[None,:])

w = np.dot(np.linalg.inv(Sw),(m1-m2)[:,None])

#ploting mapped points WT*X
y_c1_map=[]
for i in range(len(c1_p)):
    y_c1_map.append(np.dot(w.T,c1_p[i,:].T))
y_c2_map=[]
for i in range(len(c2_p)):
    y_c2_map.append(np.dot(w.T,c2_p[i,:].T))

fig = plt.figure()
plt.scatter(y_c1_map,np.zeros(100),label='C1',color='r')
plt.scatter(y_c2_map,np.zeros(100),label='C2',color='b')
plt.xlabel('x')
plt.title('Problem5c Projected Points')
plt.legend()
plt.savefig('Problem5c_projected_points.png')


# Plot the learned direction
fig = plt.figure()
plt.scatter(c1_p[:,0],c1_p[:,1],label='C1',color='r')
plt.scatter(c2_p[:,0],c2_p[:,1],label='C2',color='b')
plt.plot([-0.02,0.02],[-0.02*w[1]/w[0],0.02*w[1]/w[0]],label='w')
plt.title('Problem 5c')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.savefig('Problem_5c.png')