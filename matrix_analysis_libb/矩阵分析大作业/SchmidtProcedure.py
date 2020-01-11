# -*- coding:UTF8 -*-
import numpy as np
import sys,os
import math

from tools.loadData import load_array
from tools import ulti

path = os.getcwd()


def Schmidt_procedure(mat, m, n):
    # mat_B = mat.copy()
    Q = np.zeros((m,n))
    R = np.zeros((n,n))
    for col in range(n):
        curr_col = mat[:,col]
        # print(math.sqrt(np.sum(np.square(curr_col))))
        if col == 0:
            R[0,col] = math.sqrt(np.sum(np.square(curr_col)))
            q = curr_col / R[0,col]
            Q[:,col] = q.copy()
        else:
            q = curr_col.copy()
            for j in range(col):
                R[j,col] = np.matmul(Q[:, j], mat[:, col])
                
            for k in range(col):
                q -= R[k,col] * Q[:, k]
            R[col,col] = math.sqrt(np.sum(np.square(q)))
            q = q/ R[col,col]
            Q[:,col] = q.copy()
    return Q,R

if __name__ == "__main__":
    input_file=path+'/data/example2.txt'
    # output_file=''
    matrix, m, n = load_array(input_file,"QR")
    ulti.print_array(matrix, m, n )
    if matrix.size ==0:
        print("input Error!")
        sys.exit()
    # 首先判断矩阵是否是列线性无关的
    mat_rank = ulti.rank_of_matrix(matrix,m,n)
    if mat_rank<n:
        print("Error, the matrix with linearly dependent columns Can Not be uniquely factored as A=QR!\n\n")
        print("123")
        sys.exit()
    Q,R = Schmidt_procedure(matrix,m,n)
    print("Q=")
    ulti.print_array(Q,m,n)
    print("R=")
    ulti.print_array(R,n,n)
