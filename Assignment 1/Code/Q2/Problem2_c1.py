import numpy as np
import matplotlib.pyplot as plt

n = 50
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

plt.scatter(x_p1,x_p2,color='red',label=1)
plt.scatter(x_n1,x_n2,color='blue',label = -1)

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Problem2c Test Data')
plt.legend()
plt.savefig('2c_testdata.png')
plt.show()

x = np.asarray(x).T
y = np.asarray(y)
f =open('2c_test_data.txt','w')
for i in range(len(x[0])):
    f.write(f'{x[0,i]} {x[1,i]} {y[i]}\n')
f.close()

