import matplotlib.pyplot as plt
import numpy as np
import math

a = np.array([[0.,2.,2.,0.], [0.,0.,2.,2.]], dtype=np.float64)
b = np.array([[4.,6.,6.,4.], [4.,4.,6.,6.]], dtype=np.float64)

a_t=np.matrix(a)
b_t = np.matrix(b)
m1 = np.matrix(a.mean(axis=1)).T
m2 = np.matrix(b.mean(axis=1)).T

c1 = np.cov(a_t) / 4 * 3
c2 = np.cov(b_t) / 4 * 3
c1_i = np.linalg.inv(c1)
c2_i = np.linalg.inv(c2)
c_i =c1_i

d1 = np.matmul((m1-m2).T, c_i)
k1 = 1/2 * np.matmul(np.matmul(m1.T, c_i),m1) - 1/2 * np.matmul(np.matmul(m2.T, c_i), m2)

x = np.arange(0,7,1)
y = k1[0,0]/d1[0,1] - (d1[0,0]*x)/d1[0,1]
# print(m1,'\n\n', c1, '\n\n', c1_i)
# print(m1,'\n\n', c2, '\n\n', c2_i)
# print(d1,d1[0,0], '\n\n', k1[0,0])

plt.plot(a[0],a[1],"ro")
plt.plot(b[0], b[1], "bo")
plt.plot(x,y)
plt.show()
