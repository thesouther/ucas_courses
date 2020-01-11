# -*- coding:UTF8 -*-
import numpy as np
import sys,os
import math

from tools.loadData import load_array
from tools import ulti

path = os.getcwd()

# Housholder reduction
def Housholder_reduction(mat, m, n):
    p_factor = 1.
    P = np.identity(m)
    mat_A = mat.copy()
    for diag_idx in range(min(m,n)):
        A = mat_A[diag_idx:, diag_idx:].copy()
        if A.shape==(1,1):
            break
        # ulti.print_array(A,m-diag_idx,n-diag_idx)
        # 计算 u,R,和R的系数
        x = A[0:,0].copy()
        a1_norm =round(math.sqrt(np.sum(np.square(x))),4)
        e1 = np.zeros(m-diag_idx)
        e1[0] = 1
        I = np.identity(m-diag_idx)
        u = np.matrix(x - a1_norm * e1)
        r1_factor = 2/ np.matmul(u, u.T)[0,0]
        r1 = np.matmul(u,u.T)[0,0] * I / 2 - np.matmul(u.T, u)
        # ulti.print_array(r1, m-diag_idx,m-diag_idx)

        # 当前P矩阵，p_factor是其系数
        P_helper = np.round(np.identity(m) * np.matmul(u,u.T)[0,0] /2, 3)
        P_helper[diag_idx:, diag_idx:] = np.round(r1,3)
        p_factor *= round(r1_factor,3)
        P = np.round(np.matmul(P_helper,P),3)
        # print("P=")
        # ulti.print_array(p_factor*  P,m,m)

        # 矩阵mat_A = P*mat_A
        # A_factor = r1_factor
        A = np.round(np.matmul(r1_factor* r1, A),3)
        # print("A=")
        # ulti.print_array(A, m-diag_idx,n-diag_idx)

        # 把A复制到mat_A相应的位置
        # mat_A_factor *= A_factor
        # mat_A = np.matmul(u,u.T)[0,0] *mat_A /2
        mat_A[diag_idx:, diag_idx:] = A
        # print("mat_A=")
        # ulti.print_array(mat_A,m,n)
    P = np.round(p_factor *P, 3)
    T = mat_A #* mat_A_factor
    return P.T,T

if __name__ == "__main__":
    input_file=path+'/data/example2.txt'
    # output_file=''
    matrix, m, n = load_array(input_file,"HR")
    # ulti.print_array(matrix, m, n )
    if matrix.size ==0:
        print("input Error!")
        sys.exit()
    Q,R = Housholder_reduction(matrix,m,n)
    print("Q=")
    ulti.print_array(Q,m,m)
    print("R=")
    ulti.print_array(R,m,n)
