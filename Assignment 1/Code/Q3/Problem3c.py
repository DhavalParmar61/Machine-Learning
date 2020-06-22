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
y = np.asarray(y).astype(np.float)

f = open('Problem3c_MSEs.txt','w')
n = len(x)
x_plot = np.linspace(-1,1,100)
for k in range(1,11):
    if k==1:
        x_k = np.vstack([np.ones(n),x])
        x_plot_k = np.vstack([np.ones(n),x_plot])
    else:
        x_k = np.vstack([x_k,np.power(x,k)])
        x_plot_k = np.vstack([x_plot_k, np.power(x_plot,k)])
    x_temp = x_k.T
    w = np.linalg.inv(np.dot(x_temp.T,x_temp))
    w = np.dot(w,x_temp.T)
    w = np.dot(w,y)
    # MSE
    y_pred = np.dot(x_temp,w)
    mse = np.sum(np.power((y-y_pred),2))/n
    print (f'MSE for k={k} is {mse}')
    f.write(f'MSE for k={k} is {mse}\n')

    # For Plot
    fig = plt.figure()
    y_plot = np.dot(x_plot_k.T,w)
    plt.plot(x_plot, y_plot, label='y=wx+b', color='r')
    plt.scatter(x, y, label='Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Problem 3c  K = {k}')
    plt.legend()
    plt.savefig(f'Problem3c_k{k}.png')