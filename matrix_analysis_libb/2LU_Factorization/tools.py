# -*- coding:UTF8 -*-
import numpy as np
import sys,os
import h5py
import math

# 形式化输出数组
def print_array(mat, m,n):
    for i in range(m):
        for j in range(n):
            print("%7.6s" % mat[i,j], end=',')
        print()
    print()

# 从文件中导入数组
def load_array(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        mat = []
        i = 1
        for line in lines:
            line = line.strip().split(' ')
            line = list(map(eval, line))
            line.append(i)
            i += 1
            mat.append(line)
        file.close()
        mat = np.array(mat, dtype=np.float64)
        m,n = mat.shape
        if m != n-1:
            print("the shape of matrix is wrong")
            return np.array([])
        return mat, m, n

# 从文件中load数组并显示
def display_result(output_file):
    h_file = h5py.File(output_file, 'r')
    for i in h_file.keys():
        print(h_file[i].value)
    h_file.close()