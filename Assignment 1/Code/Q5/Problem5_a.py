import numpy as np
import matplotlib.pyplot as plt

c1 = np.random.uniform(low=-1,high=1,size=100)
c2 = np.hstack([np.random.uniform(low=-3,high=-2,size=50),np.random.uniform(low=2,high=3,size=50)])

f = open('data_c1.txt','w')
for i in range(len(c1)):
    f.write(f'{c1[i]}\n')
f.close()

f = open('data_c2.txt','w')
for i in range(len(c2)):
    f.write(f'{c2[i]}\n')
f.close()

# plot data
fig = plt.figure()
plt.scatter(c1,np.zeros(100),label='C1',color='r')
plt.scatter(c2,np.zeros(100),label='C2',color='b')
plt.title('Problem 5a')
plt.xlabel('x')
plt.legend()
plt.savefig('Problem_5a.png')