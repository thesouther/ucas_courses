# -*- coding:UTF8 -*-
import numpy as np
# import h5py

# 形式化输出数组
def print_array(mat, m,n):
    for i in range(m):
        for j in range(n):
            print("%7.6s" % mat[i,j], end=',')
        print()
    # print()

# # 从文件中load数组并显示
# def display_result(output_file):
#     h_file = h5py.File(output_file, 'r')
#     for i in h_file.keys():
#         print(h_file[i].value)
#     h_file.close()

# 计算矩阵的秩
def rank_of_matrix(mat, m, n):
    # 使用部分消元法进行初等行变换
    mat=mat.copy()
    row=col=0
    while row<m-1 and col<n:
        curr_col = np.abs(mat[row:,col])
        max_item = np.max(curr_col)
        # print(curr_col, max_item)
        if np.all(curr_col == 0):
            col+=1
            continue
        if curr_col[0] != max_item: # 行交换
            max_row_idx = np.where(curr_col == max_item)[0][0] + row
            helper = mat[max_row_idx].copy()
            mat[max_row_idx] = mat[row]
            mat[row] = helper
        # print_array(mat, m, n)
        # print()
        
        for i in range(row+1, m):
            if np.all(mat[i]==0):
                break
            factor = mat[i,col] / mat[row,col]
            value = (-1 * factor * mat[row, col:])
            mat[i, col:] += value
        col += 1 
        row += 1
        # print_array(mat,m,n) 
        # print()

    zero_rows=0
    for i in mat[::-1]:
        if np.all(i==0):
            zero_rows += 1
            continue
        break
    mat_rank = min(m,n,m-zero_rows)
    # print(m,n,zero_rows,mat_rank)
    return mat_rank

