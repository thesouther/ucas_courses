#user/bin/python
# -*- coding:UTF-8 -*-

import numpy as np
import math

def cal_SVD(matr, m, n):
    # print(m,n)
    CC_T = np.matmul(matr, matr.T)
    C_TC = np.matmul(matr.T, matr)
    # print(CC_T)
    # print(C_TC)
    eigenValuesofCC_T = np.linalg.eig(CC_T)
    eigenValuesofC_TC = np.linalg.eig(C_TC)
    sigma_matr = np.sqrt(np.diag(eigenValuesofC_TC[0]))
    U = eigenValuesofCC_T[1].T
    V = eigenValuesofC_TC[1].T
    print(U)
    print(V)
    print(sigma_matr)
    # print(np.matmul(np.matmul(U,sigma_matr), V.T))

if __name__ == "__main__":
    matr = np.array([[1,0,1,0,0,0],
                     [0,1,0,0,0,0],
                     [1,1,0,0,0,0],
                     [1,0,0,1,1,0],
                     [0,0,0,1,0,1]])
    # print(matr.shape[0])
    # cal_SVD(matr, matr.shape[0], matr.shape[1])
    u,sigma,vt=np.linalg.svd(matr)
    print(u)
    print(sigma)
    print(vt)