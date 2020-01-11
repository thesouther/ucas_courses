# -*- coding:UTF8 -*-
import numpy as np
import sys,os
import math

from tools.loadData import load_array
from tools import ulti

path = os.getcwd()

# Housholder reduction
def Givens_reduction(mat, m, n):
    P = np.identity(m)
    mat_A = mat.copy()
    for diag_idx in range(min(m,n)):
        curr_col = mat_A[diag_idx+1:, diag_idx]
        # print(curr_col)
        if len(curr_col) == 0:
            break
        for i in range(len(curr_col)):
            if round(curr_col[i],3) == 0. :
                continue
            P_helper = np.identity(m)
            cs_factor = math.sqrt(mat_A[diag_idx,diag_idx]**2 + curr_col[i]**2)
            # print(mat_A[diag_idx,diag_idx]**2 + curr_col[i]**2)
            # 计算旋转矩阵
            P_helper[diag_idx,diag_idx] = mat_A[diag_idx,diag_idx] / cs_factor
            P_helper[diag_idx, diag_idx + i+1] = curr_col[i] / cs_factor
            P_helper[diag_idx+i+1, diag_idx] = - curr_col[i] / cs_factor
            P_helper[diag_idx+i+1, diag_idx+i+1] = mat_A[diag_idx,diag_idx] / cs_factor
            # print(i,P_helper)
            # 计算某一位置零后的mat_A
            P = np.round(np.dot(P_helper, P),4)
            mat_A = np.round(np.dot(P_helper, mat_A),4)
        # print("mat_A=")
        # ulti.print_array(mat_A,m,n)
    Q = P.T
    T=mat_A
    return Q,T

if __name__ == "__main__":
    input_file=path+'/data/example1.txt'
    # output_file=''
    matrix, m, n = load_array(input_file,"HR")
    # ulti.print_array(matrix, m, n )
    if matrix.size ==0:
        print("input Error!")
        sys.exit()
    Givens_reduction(matrix,m,n)
    Q,R = Givens_reduction(matrix,m,n)
    print("Q=")
    ulti.print_array(Q,m,m)
    print("R=")
    ulti.print_array(R,m,n)
