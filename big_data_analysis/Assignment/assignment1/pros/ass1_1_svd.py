#！user/bin/python
# -*- coding:UTF-8 -*-

'''
file_name: 大数据分析作业1-1，
'''
import os, path
from scipy.sparse import csr_matrix
import gzip
import matplotlib.pyplot as plt
import scipy.sparse.linalg as slin
import numpy as np

def load_file(file_name, mode):
    f = gzip.open(file_name, mode)
    return f

# 生成行坐标、列坐标、数据值
def gen_csr_matrix(file_name):
    xs = []
    ys = []
    data =[]

    with load_file(file_name, 'rb') as file_data:
        datas = file_data.readlines()
        for line in datas:
            line = line.strip().decode().split(' ')
            xs.append(int(line[0]))
            ys.append(int(line[1]))
            data.append(1)
        file_data.close()

    m = max(xs) + 1
    n = max(ys) + 1
    # print( m,n,len(data))
    Mat = csr_matrix((data, (xs, ys)), shape=(m, n),dtype=np.float64)
    return Mat

# 画原始数据的分布图
def plt_ori_data(Mat):
    idxs = Mat.todok().tocoo()
    rows = list(idxs.row.reshape(-1))
    cols = list(idxs.col.reshape(-1))
    # print(len(row))

    plt.plot(rows, cols)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    plt.show()

#画出10副图
def plt_u_gra(U, v):
    plt.figure(figsize=(10,10))
    for i in range(0,10,2):
        plt.subplot(5, 2, 1 + int(i/2))
        plt.plot(U[:,i], U[:,i+1])
        plt.xlabel('u'+str(i))
        plt.ylabel('u'+str(i+1))
        plt.title('u'+str(i) + ' - u'+str(i+1), color='blue', fontweight='bold', verticalalignment='bottom')
        plt.xticks([-0.3,0.3])
        plt.yticks([-0.3,0.3])

    for j in range(0,10,2):
        plt.subplot(5, 2, 6 + int(j/2))
        plt.plot(v[j], v[j+1])
        plt.xlabel('v'+str(j))
        plt.ylabel('v'+str(j+1))
        plt.title('v'+str(j) + ' - v'+str(j+1), color='blue', fontweight='bold', verticalalignment='bottom')
        plt.xticks([-0.7,0.7])
        plt.yticks([-0.7,0.7])
        # plt.tight_layout()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_name = "./pros/data/yelp.edgelist.gz"
    Mat = gen_csr_matrix(file_name)
    # plt_ori_data(Mat)
    Mat = Mat.asfptype()
    u, sigma, vt = slin.svds(Mat,k=10,which='LM')
    # print(np.array(u)[:,0].shape)
    # print(u.shape)
    plt_u_gra(u, vt)
    
    