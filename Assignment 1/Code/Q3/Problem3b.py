import numpy as np
import matplotlib.pyplot as plt

f = open('train_data.txt','r')
x= []
y = []
for line in f :
    x_temp,y_temp = line.split(' ')
    x.append(x_temp)
    y.append(y_temp)
x = np.asarray(x).astype(np.float)
y = np.asarray(y).astype((np.float))

x_mean = np.sum(x)/len(x)
y_mean = np.sum(y)/len(y)

w = np.sum((x-x_mean)*(y-y_mean))/np.sum(np.power((x-x_mean),2))
b = y_mean - w*x_mean

#MSE
y_pred = np.asarray([(w*i+b) for i in x])
mse = np.sum(np.power((y-y_pred),2))/len(x)
print(f'Mean Square Error : {mse}')

#Plot the Graph
x_line = np.linspace(-1,1,100)
y_line = np.asarray([(w*i+b) for i in x_line])

plt.plot(x_line,y_line,label='y=wx+b',color='r')
plt.scatter(x,y,label='Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Problem 3b')
plt.legend()
plt.show()

