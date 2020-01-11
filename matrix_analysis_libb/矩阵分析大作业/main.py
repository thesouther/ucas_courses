# -*- coding:UTF8 -*-
import numpy as np
import os,sys
import argparse
from tools import loadData
from tools import ulti
from LUFactorization import LU_factorization
from SchmidtProcedure import Schmidt_procedure
from HouseholderReduction import Housholder_reduction
from GivensReduction import Givens_reduction

# 获取当前工作路径
path = os.getcwd()

ap = argparse.ArgumentParser(description="Four Reduction Methods")
#选择使用的约简模型，四个选项'LU','QR','HR','GR'，其中LU代表LU factorization, QR代表Schmidt Procedure, HR代表Housholder Reduction, GR代表Givens Reduction!
ap.add_argument("--model", type=str, choices=['LU','QR','HR','GR'], default="LU",
                help="4 choices in ['LU','QR','HR','GR'], LU->LU factorization, QR->Schmidt Procedure, HR->Housholder Reduction, GR->Givens Reduction!")

ap.add_argument("--input", type=str, default="data/example1.txt",
                help="input example file path.")
# ap.add_argument("--egnum", type=int, default=1,
#                 help="number of examples to test!")

args = ap.parse_args()


if __name__ == "__main__":
    input_file=path+'/'+args.input
    # print(input_file)
    matrix, m, n= loadData.load_array(input_file, args.model)
    if matrix.size ==0:
        print("input Error!")
        sys.exit()

    if args.model == "LU":
        print("\nBe careful you have selected LU Factorization, the input should be a square matrix.\n")
    elif args.model == "QR":
        # 首先判断矩阵是否是列线性无关的
        mat_rank = ulti.rank_of_matrix(matrix,m,n)
        if mat_rank<n:
            print("Error!\nThe matrix with linearly dependent columns Can Not be uniquely factored as A=QR!\n\n")
            sys.exit()

    print("="*50,"\norigin matrix type: {m} * {n}" .format(m=m,n=n),"\nOrigin Matrix A = ")
    ulti.print_array(matrix, m, n)
    print("="*50, "\nProcessing:")
    if args.model == "LU":
        L,U,P = LU_factorization(matrix, m, n)
        print("L=")
        ulti.print_array(L,m,m)
        print("U=")
        ulti.print_array(U,m,m)
        print("P=")
        ulti.print_array(P,m,m)
    elif args.model == "QR":
        Q,R = Schmidt_procedure(matrix,m,n)
        print("Q=")
        ulti.print_array(Q,m,n)
        print("R=")
        ulti.print_array(R,n,n)
    elif args.model == "HR":
        Q,R = Housholder_reduction(matrix,m,n)
        print("Q=")
        ulti.print_array(Q,m,m)
        print("R=")
        ulti.print_array(R,m,n)
    elif args.model=="GR":
        Q,R = Givens_reduction(matrix,m,n)
        print("Q=")
        ulti.print_array(Q,m,m)
        print("R=")
        ulti.print_array(R,m,n)