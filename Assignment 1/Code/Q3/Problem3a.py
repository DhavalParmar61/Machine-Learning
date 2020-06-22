import numpy as np
import matplotlib.pyplot as plt

def get_data(n,view_plt=False):
    f = open('test_data.txt','w')
    y = []
    x = []
    for i in range(n):
        x_temp=np.random.uniform(low=-1,high=1,size=1)
        x.append(x_temp)
        y.append(np.sin(3*x_temp))
        f.write(f'{x_temp[0]} {y[i][0]}\n')
    f.close
    if view_plt:
        plt.scatter(x,y,label='y=sin(3x)')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Problem 3a Test Data')
        plt.legend()
        plt.show()

    x = np.asarray(x)
    y = np.asarray(y)
    return x,y

get_data(100,True)