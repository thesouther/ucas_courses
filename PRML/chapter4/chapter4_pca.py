# -*- coding:UTF-8 -*-
import numpy as np
import sys, os
import matplotlib.pyplot as plt

# 求协方差矩阵
def c_x(mat):
    mx = mat.sum(axis=0)/len(mat)
    print(mat.sum(axis=0)/len(mat))
    mt = mat - [mx for _ in range(len(mat))]
    mm = np.zeros(shape=(3,3))
    for i in mt:
        mult = np.matrix(i).T * np.matrix(i)
        mm += mult
    # print(mm/len(mat))
    return mm / len(mat)

def des_dim(x,eig_vector, dim):
    phi = np.matrix([eig_vector[i] for i in range(dim)])
    # print(phi)
    desed_dim = phi * np.matrix(x)
    # print(desed_dim)
    return desed_dim

def plot_dot(x):
    plt.figure()
    plt.plot(np.array(x[0])[0,0:4], np.array(x[1])[0,0:4],'ro',label="w1")
    plt.plot(np.array(x[0])[0,4:8], np.array(x[1])[0,4:8],'bx', label="w2")
    plt.legend(numpoints=1)
    plt.show()

if __name__ == "__main__":
    # mat=np.array([[0,0,0],
    # [2,0,0],
    # [2,0,1],
    # [1,2,0],
    # [0,0,1],
    # [0,1,0],
    # [0,-2,1],
    # [1,1,-2]])
    mat=np.array([
    [-0.75,-0.25,-0.125],
    [1.25,-0.25,-0.125],
    [1.25,-0.25,0.875],
    [0.25,1.75,-0.125],
    [-0.75,-0.25,0.875],
    [-0.75,0.75,-0.125],
    [-0.75,-2.25,0.875],
    [0.25,0.75,-2.125]])
    # mat2=np.array([[0,0,0],
    # [1,0,0],
    # [1,0,1],
    # [1,1,0],
    # [0,0,1],
    # [0,1,0],
    # [0,1,1],
    # [1,1,1]])
    cx = c_x(mat)
    print(cx)
    eigen_value = np.linalg.eig(cx)
    print(eigen_value[0])
    print(eigen_value)
    print(np.matrix(eigen_value[1]).T)

    # 降维
    desed_dim2 = des_dim(mat.T, eigen_value[1],2)
    print(desed_dim2.T)
    plot_dot(desed_dim2)

    desed_dim1 = des_dim(mat.T, eigen_value[1],1)
    print(desed_dim1)
    plt.plot(np.array(desed_dim1[0])[0,0:4],[0 for i in range(-2,2)],'ro', label="w1")
    plt.plot(np.array(desed_dim1[0])[0,4:8],[0 for i in range(-2,2)],'bx', label="w2")
    plt.legend(numpoints=1)
    plt.show()
