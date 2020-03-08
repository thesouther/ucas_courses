import numpy as np
import matplotlib.pyplot as plt

# 生成数据
def gen_data(k=3,data_num=1500):
    unit_count = int(data_num/k)
    mean1 = [0,0]
    mean2 = [10,20]
    mean3 = [20,10]
    cov1 = [[1,0],[0,10]]
    cov2 = [[15,5],[5,5]]
    cov3 = [[5,5],[5,15]]
    data1 = np.random.multivariate_normal(mean1,cov1,unit_count)
    data2 = np.random.multivariate_normal(mean2,cov2,unit_count)
    data3 = np.random.multivariate_normal(mean3,cov3,unit_count)
    y1 = np.array([0 for i in range(unit_count)])
    y2 = np.array([1 for i in range(unit_count)])
    y3 = np.array([2 for i in range(unit_count)])
    if k==2:
        data = np.concatenate((data1,data2))
        y = np.concatenate((y1,y2))
    else:
        data = np.concatenate((data1,data2,data3))
        y = np.concatenate((y1,y2,y3))
    return data,y

# 展示数据
def show_data(data):
    data = data.T
    plt.scatter(data[0],data[1])
    plt.axis()
    plt.title("test_data")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


class GMM:
    def __init__(self, K, max_iter=200, mu=None, sigma=None, alpha=None, gamma=None, epsilon=1e-6):
        self.K=K
        self.max_iter = max_iter
        self.mu = mu
        self.sigma= sigma
        self.alpha = alpha
        self.gamma=gamma
        self.epsilon = epsilon

    def gaussian(self, x, mu, sigma):
        # 数据特征维度
        dim = np.shape(sigma)[0]
        # 计算协方差矩阵的行列式和逆
        sigma_dets = np.linalg.det(sigma + np.eye(dim)*0.001)
        sigma_inv = np.linalg.inv(sigma + np.eye(dim)*0.001)
        #计算exp的平方项
        x_diff = (x-mu).reshape((1,dim))
        exp_power = -0.5* x_diff.dot(sigma_inv).dot(x_diff.T)[0][0]
        # 计算高斯概率密度
        prob_normal = 1.0 / np.power(np.power(2*np.pi,dim)*np.abs(sigma_dets),0.5) * np.exp(exp_power)
        # print(prob_normal)
        return prob_normal

    def fit(self, X):
        # 样本数目和特征维数
        X_size, X_dim = np.shape(X)
        # 初始化参数，包括均值mu、协方差矩阵sigma、隶属度系数alpha、隐变量gamma
        if self.mu is None:
            self.mu = []
            for i in range(self.K):
                mean = np.mean(X,axis=0)
                self.mu.append(mean)
        if self.sigma is None:
            self.sigma=[]
            for i in range(self.K):
                cov = np.cov(X,rowvar=False)
                self.sigma.append(cov)
        self.alpha = np.random.rand(self.K)
        self.alpha /= self.alpha.sum()
        self.alpha =np.round(self.alpha,5)
        self.gamma = [np.zeros(self.K) for i in range(X_size)]

        old_mu = []
        for iter in range(self.max_iter):
            old_mu=self.mu.copy()
            # E步，计算期望
            p_x = [[self.alpha[k]*self.gaussian(X[j],self.mu[k], self.sigma[k]) for k in range(self.K)] for j in range(X_size)]
            sum_px = np.sum(p_x,axis=1)
            self.gamma = [p_x[i]/sum_px[i] for i in range(X_size)]
            # M步
            # 计算每个样本的隶属度
            sum_gamma = np.sum(self.gamma,axis=0)
            self.alpha = (1.0*sum_gamma) / X_size
            for k in range(self.K):
                # 计算均值
                x_mult_gamma = [self.gamma[i][k]*X[i] for i in range(X_size)]
                self.mu[k] = np.sum(x_mult_gamma,axis=0) / sum_gamma[k]
                # 计算协方差
                X_norm = [X[i]-self.mu[k] for i in range(X_size)]
                x_diff = [(self.gamma[i][k]* X_norm[i].reshape(X_dim,1).dot(X_norm[i].reshape(1,X_dim))).tolist() for i in range(X_size)]
                self.sigma[k] = list(np.sum(x_diff,axis=0)) / sum_gamma[k]
            # print(old_mu,self.mu)
            # 提起那结束条件
            if np.max(np.abs([self.mu[i] - old_mu[i] for i in range(X_dim)]))<self.epsilon:
                break
        self.prediction = [np.argmax(self.gamma[i]) for i in range(X_size)]
        return self.alpha,self.mu,self.sigma


if __name__ == "__main__":
    data_num = 1500
    k=3
    X,y = gen_data(k=k,data_num=data_num)
    # 展示数据分布
    # show_data(X)
    idx = int(data_num/k)
    mu_init = [np.mean(X[0*idx:1*idx],axis=0),np.mean(X[1*idx:2*idx],axis=0),np.mean(X[2*idx:3*idx],axis=0)]
    sigma_init = [np.cov(X,rowvar=False),np.cov(X,rowvar=False),np.cov(X,rowvar=False)]
    gmm = GMM(K=3,mu=mu_init,sigma=sigma_init)
    gmm.fit(X)
    prediction = gmm.prediction
    acc = 0
    for i in range(data_num):
        if prediction[i] == y[i]: acc+=1
    acc /=data_num
    print("="*20, "Means of data:", "="*20)
    for i in np.round(gmm.mu,3):
        print(i)

    print('\n',"="*20, "Covariances of data:", "="*20)
    for i in np.round(gmm.sigma,3):
        print(i)

    print("\nAccuracy: {}% \n".format(acc*100))
