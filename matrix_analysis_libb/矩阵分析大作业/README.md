# 矩阵分析大作业


本项目为矩阵分析与应用课程课后大作业，具体包括矩阵的LU分解、QR分解（Gram-Schmidt）、正交约简 (Householder reduction 和Givens reduction)四个部分。
## 环境
* python 3.7
* numpy 1.15.1

### 文件描述:
```
FinalAssignment
|____ data                          # 样例文件夹，里边包含测试用的矩阵，包括一个方阵（example1）和一个长方形阵（example2）
|____ tools                         # 工具类文件，包括一些读写文件、格式化输出等的函数
|____ main.py                       # 运行四种约简过程的主函数
|____ LUFactorization.py            # LU分解过程脚本
|____ SchmidtProcedure.py           # Schmidt过程函数脚本
|____ HousholderReduction.py        # Housholder约简函数脚本
|____ GivensReduction.py            # Givens约简函数脚本
|____ run.sh                        # bash脚本

```
### 运行
如果时Linux系统或者mac系统，可以直接运行bash文件
```
bash run.sh
```

或者运行python脚本

```
python3 main.py -h
```

将会有如下提示:

```
usage: main.py [-h] [--model {LU,QR,HR,GR}] [--input INPUT]

Four Reduction Methods

optional arguments:
  -h, --help            show this help message and exit
  --model {LU,QR,HR,GR}
                        4 choices in ['LU','QR','HR','GR'], LU->LU
                        factorization, QR->Schmidt Procedure, HR->Housholder
                        Reduction, GR->Givens Reduction!
  --input INPUT         input example file path.
```

**注意**： 运行时分解过程的选项有四种，各选项代表的含义为：
  - `--model=LU` : factorization
  - `--model=QR` : Schmidt Procedure
  - `--model=HR` : Housholder Reduction
  - `--model=GR` : Givens Reduction 

例如运行
> python .\main.py --model=LU --input=data/example1.txt
<!-- 1. 本程序可运行多个例子，支持在.data/example.txt中使用多个矩阵实例，并通过在运行是使用参数'--egnum'(default=1)进行指定需要运行的实例个数。 -->

#### 结果
```

Be careful you have selected LU Factorization, the input should be a square matrix.

==================================================
origin matrix type: 4 * 5
Origin Matrix A =
    1.0,    2.0,   -3.0,    4.0,    1.0,
    4.0,    8.0,   12.0,   -8.0,    2.0,
    2.0,    3.0,    2.0,    1.0,    3.0,
   -2.0,   -1.0,    1.0,   -4.0,    4.0,

==================================================
Result:
L=
    1.0,    0.0,    0.0,    0.0,
   -0.5,    1.0,    0.0,    0.0,
   0.25,    0.0,    1.0,    0.0,
    0.5, -0.333, 0.2777,    1.0,
U=
    4.0,    8.0,   12.0,   -8.0,
    0.0,    3.0,    7.0,   -8.0,
    0.0,    0.0,   -6.0,    6.0,
    0.0,    0.0,    0.0, 0.6666,
P=
    0.0,    1.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    1.0,
    1.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    1.0,    0.0,
```

### 注意

此程序中包含对输入矩阵合理性的验证，包括：
  - 当需要使用LU分解时，输入需要时方阵；
  - 当需要运行Schmidt-QR分解过程时，输入的矩阵应当是列线性无关的。


---
## 运行实例及结果

> python .\main.py --model=LU --input=data/example1.txt

```

Be careful you have selected LU Factorization, the input should be a square matrix.

================================================== 
origin matrix type: 4 * 5 
Origin Matrix A = 
    1.0,    2.0,   -3.0,    4.0,    1.0,
    4.0,    8.0,   12.0,   -8.0,    2.0,
    2.0,    3.0,    2.0,    1.0,    3.0,
   -2.0,   -1.0,    1.0,   -4.0,    4.0,

==================================================
Result:
L=
    1.0,    0.0,    0.0,    0.0,
   -0.5,    1.0,    0.0,    0.0,
   0.25,    0.0,    1.0,    0.0,
    0.5, -0.333, 0.2777,    1.0,
U=
    4.0,    8.0,   12.0,   -8.0,
    0.0,    3.0,    7.0,   -8.0,
    0.0,    0.0,   -6.0,    6.0,
    0.0,    0.0,    0.0, 0.6666,
P=
    0.0,    1.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    1.0,
    1.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    1.0,    0.0,
```

> python .\main.py --model=QR --input=data/example1.txt

```
================================================== 
origin matrix type: 4 * 4 
Origin Matrix A =
    1.0,    2.0,   -3.0,    4.0,
    4.0,    8.0,   12.0,   -8.0,
    2.0,    3.0,    2.0,    1.0,
   -2.0,   -1.0,    1.0,   -4.0,
==================================================
Processing:
Q=
    0.2, 0.1173, -0.940, -0.247,
    0.8, 0.4692, 0.2905, -0.235,
    0.4, -0.131, -0.166, 0.8916,
   -0.4, 0.8652, -0.055, 0.2972,
R=
    5.0,    8.4, 9.4000, -3.600,
    0.0, 2.7276, 5.8805, -6.877,
    0.0,    0.0, 5.9210, -6.031,
    0.0,    0.0,    0.0, 0.5944,
```

> python .\main.py --model=HR --input=data/example1.txt

```
================================================== 
origin matrix type: 4 * 4 
Origin Matrix A =
    1.0,    2.0,   -3.0,    4.0,
    4.0,    8.0,   12.0,   -8.0,
    2.0,    3.0,    2.0,    1.0,
   -2.0,   -1.0,    1.0,   -4.0,
==================================================
Processing:
Q=
  0.203,  0.119, -0.953,  0.251,
   0.81,  0.475,  0.294,  0.238,
  0.405, -0.134, -0.168, -0.903,
 -0.405,  0.876, -0.056, -0.301,
R=
    5.0,    8.4,    9.4,   -3.6,
    0.0,  2.728,   5.88, -6.878,
    0.0,    0.0,  5.921, -6.032,
    0.0,   -0.0,    0.0, -0.594,
```

> python .\main.py --model=GR --input=data/example1.txt

```
================================================== 
origin matrix type: 4 * 4 
Origin Matrix A =
    1.0,    2.0,   -3.0,    4.0,
    4.0,    8.0,   12.0,   -8.0,
    2.0,    3.0,    2.0,    1.0,
   -2.0,   -1.0,    1.0,   -4.0,
==================================================
Processing:
Q=
    0.2, 0.1173, -0.940, -0.247,
 0.7999, 0.4692, 0.2905, -0.235,
    0.4, -0.131, -0.166, 0.8915,
   -0.4, 0.8652, -0.055, 0.2972,
R=
    5.0,    8.4,    9.4,   -3.6,
    0.0, 2.7276, 5.8805, -6.877,
    0.0,    0.0, 5.9211, -6.031,
    0.0,    0.0,    0.0, 0.5945,
```