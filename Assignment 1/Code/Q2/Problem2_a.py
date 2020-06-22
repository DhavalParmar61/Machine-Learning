import numpy as np
import matplotlib.pyplot as plt


n=50
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
    y.append(np.sign(np.dot(w,x_temp)+b))
    if y[i]>0:
        x_p1.append(x_temp[0])
        x_p2.append(x_temp[1])
    else:
        x_n1.append(x_temp[0])
        x_n2.append(x_temp[1])


plt.scatter(x_p1,x_p2,color='red',label=1)
plt.scatter(x_n1,x_n2,color='blue',label = -1)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Problem 2a Testdata')
plt.legend()
plt.savefig('Problem2a_Testdata.png')

x = np.asarray(x).T
y = np.asarray(y)
f =open('2a_test_data.txt','w')
for i in range(len(x[0])):
    f.write(f'{x[0,i]} {x[1,i]} {y[i]}\n')
f.close()