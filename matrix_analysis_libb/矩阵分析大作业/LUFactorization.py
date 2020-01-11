# -*- coding:UTF8 -*-
import numpy as np
import sys,os
import h5py
import math

from tools import loadData
from tools import ulti

path = os.getcwd()

# 进行LU分解，（可选，将分解过程生成的矩阵存到文件里）
def LU_factorization(mat, m, n):
    '''
    如果想保存成文件，取消这段注释
    # h_file = h5py.File(output_file, 'w')
    arr_idx = 0
    '''
    # 对使用行标进行增广的矩阵，使用部分消元法进行初等行变换
    mat_B = mat.copy()
    for row in range(m-1):
        curr_col = np.abs(mat[row:,row])
        max_item = np.max(curr_col)
        # print(curr_col, max_item)
        if curr_col[0] != max_item:
            max_row_idx = np.where(curr_col == max_item)[0][0] + row
            mat[row] = mat_B[max_row_idx]
            mat[max_row_idx] = mat_B[row]
        ulti.print_array(mat, m, n)
        print()

        '''
        h_file.create_dataset(str(arr_idx), data=mat)
        arr_idx += 1
        '''
        for i in range(row+1, m):
            factor = mat[i,row] / mat[row,row]
            mat[i, row] = factor
            value = (-1 * factor * mat[row, row+1:-1])
            mat[i, row+1:-1] += value

        mat_B = mat.copy()
        ulti.print_array(mat,m,n) 
        print()
    '''
        如果想保存成文件，取消这段注释
        h_file.create_dataset(str(arr_idx), data=mat)
        arr_idx += 1
    # h_file.close()
    '''

    print("="*50)
    # 得到上三角阵和下三角阵，并输出
    U = np.triu(mat[:, :-1],0)
    print("Result:")
    for k in range(m):
        mat_B[k,k] =1
    L = np.tril(mat_B[:,:-1], 0)
    P = np.zeros(shape=(m,n-1))
    for idx in range(m):
        row_idx = int(mat[idx, -1])
        P[idx, row_idx-1] = 1
    return L,U,P


if __name__ == "__main__":
    input_file=path+'/data/example1.txt'
    output_file=''
    # print(input_file)
    mat, m, n = loadData.load_array(input_file,"LU")
    # print(mat)
    if mat.size != 0:
        LU_factorization(mat, m, n)
        # display_result(output_file)
