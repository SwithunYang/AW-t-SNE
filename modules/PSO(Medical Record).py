import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.metrics import silhouette_score ,calinski_harabasz_score,davies_bouldin_score
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
#分割训练集与测试集
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score,precision_score,f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def standard(X):
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    xmaxmin = xmax-xmin
    n, m = X.shape
    for i in range(n):
        for j in range(m):
            X[i,j] = (X[i,j]-xmin[j])/xmaxmin[j]
    return X
data = pd.read_csv("handled_medical record_data(2000).csv")
def cal_pairwise_dist(data,weight):
    #计算pairwise 距离, x是matrix,(a-b)^2 = a^2 + b^2 - 2*a*b
    y = np.square(data) * weight
    sum_x = np.sum(y, 1)  # 计算每个数据向量的平方和
    dist = np.add(np.add(-2 * np.dot(data * weight, data.T), sum_x).T, sum_x)
    return dist


def cal_perplexity(dist, idx=0, beta=1.0):
    '''计算perplexity, D是距离向量，
    idx指dist中自己与自己距离的位置，beta是高斯分布参数
    这里的perp仅计算了熵，方便计算
    '''
    prob = np.exp(-dist * beta)  #以e为底，以-|Xi-Xj|的平方*beta为指数
    # 设置自身prob为0
    prob[idx] = 0
    sum_prob = np.sum(prob)  #所有高维空间的点Xj到Ｘi的距离的和
    perp = np.log(sum_prob) + beta * np.sum(dist * prob) / sum_prob #计算困惑度
    prob /= sum_prob #计算Pi|j
    return perp, prob


def seach_prob(x,weight, tol=1e-5, perplexity=30.0):
    '''二分搜索寻找beta,并计算pairwise的prob
    '''

    # 初始化参数
    # print("Computing pairwise distances...")
    (n, d) = x.shape
    dist = cal_pairwise_dist(x,weight)  #得到距离
    pair_prob = np.zeros((n, n))# n*n全为0的矩阵
    beta = np.ones((n, 1)) #n行1列全为1的矩阵
    base_perp = np.log(perplexity) #预先设置的困惑度取log，方便后续计算

    #所有的点
    for i in range(n):
        # if i % 500 == 0:
        #     print("Computing pair_prob for point %s of %s ..." %(i,n))

        betamin = -np.inf #无穷大
        betamax = np.inf #无穷小
        perp, this_prob = cal_perplexity(dist[i], i, beta[i])

        # 二分搜索,寻找最佳sigma下的prob
        perp_diff = perp - base_perp
        tries = 0 #更新beta迭代次数
        while np.abs(perp_diff) > tol and tries < 50:
            if perp_diff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    if beta[i] >= 100000:
                        beta[i] = beta[i] * 1.5
                    else:
                        beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    if beta[i] >= 100000:
                        beta[i] = beta[i] / 1.5
                    else:
                        beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # 更新perb,prob值
            perp, this_prob = cal_perplexity(dist[i], i, beta[i])
            perp_diff = perp - base_perp
            tries = tries + 1
        # 记录prob值
        pair_prob[i,] = this_prob
    # print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))#开方，求平均
    return pair_prob

def P_cal(x,perplexity=30.0):
    P = seach_prob(x, 1e-5, perplexity)
    P = P + np.transpose(P)        ###这里加出来的矩阵有时会出现NAN
    P = P / np.sum(P)
    P = P * 4
    P = np.maximum(P, 1e-12)
    return P
def tsne(x, no_dims,weight ,initial_dims=50, perplexity=30.0, max_iter=1000):
    """Runs t-SNE on the dataset in the NxD array x
    to reduce its dimensionality to no_dims dimensions.
    The syntaxis of the function is Y = tsne.tsne(x, no_dims, perplexity),
    where x is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array x should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # 初始化参数和变量
    #x = pca(x, initial_dims).real
    (n, d) = x.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 200
    min_gain = 0.01
    np.random.seed(2)
    y = np.random.randn(n, no_dims)
    dy = np.zeros((n, no_dims))
    iy = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # 对称化
    P = seach_prob(x, weight,1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    # early exaggeration
    P = P * 4
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):
        # Compute pairwise affinities
        sum_y = np.sum(np.square(y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))
        num[range(n), range(n)] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dy[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (y[i,:] - y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dy > 0) != (iy > 0)) + (gains * 0.8) * ((dy > 0) == (iy > 0))
        gains[gains < min_gain] = min_gain
        iy = momentum * iy - eta * (gains * dy)
        y = y + iy
        y = y - np.tile(np.mean(y, 0), (n, 1))
        # Compute current value of cost function
        if (iter + 1) % 100 == 0:
            if iter > 100:
                C = np.sum(P * np.log(P / Q))
            else:
                C = np.sum( P/4 * np.log( P/4 / Q))
            print("Iteration ", (iter + 1), ": error is ", C)
        # Stop lying about P-values
        if iter == 100:
            P = P / 4
    # print(Q)
    # np.savetxt(r'MQ.text',
    #            Q, delimiter=',')
    print("finished training!")
    return y

class PSO(object):
    def __init__(self, population_size, max_steps):
        self.w = 0.7  # 惯性权重
        self.c1 = self.c2 = 2
        self.population_size = population_size  # 粒子群数量
        self.dim = 13  # 搜索空间的维度
        self.max_steps = max_steps  # 迭代次数
        self.x_bound = [0, 1]  # 解空间范围
        # self.x = np.random.uniform(self.x_bound[0], self.x_bound[1],
        #                            (self.population_size, self.dim))  # 初始化粒子群位置
        # self.v = np.random.rand(self.population_size, self.dim)  # 初始化粒子群速
        ###初始化粒子速度可能存在问题
        self.v = np.random.uniform(0.01, 0.01,
                                   (self.population_size, self.dim))
        fitness = self.calculate_fitness(x)
        self.p = x  # 个体的最佳位置
        self.pg = x[np.argmax(fitness)]  # 全局最佳位置     argmax改为max
        self.individual_best_fitness = fitness  # 个体的最优适应度
        self.global_best_fitness = np.max(fitness)  # 全局最佳适应度

    def calculate_fitness(self,x):
        # print(np.size(x,0))
        # qdata=np.loadtxt('MQ1.text',delimiter=',')
        # data = pd.read_csv('../Project1/data.csv')
        global data
        # Scaler = StandardScaler().fit(data)
        # data = Scaler.transform(data)
        data = np.array(data)
        # data = data[:1000, :]  # 修改数据量
        fit=[]
        ###x是PSO空间生成的点（也就是权重点）
        for i in range(np.size(x,0)):
            x[i]=x[i]/np.sum(x[i])
            # print(x[i])
            # P=P_cal(X,40.0)
            data_Tsne = tsne(data, 2, x[i], 50, 40.0)
            model = KMeans(n_clusters=18, init='k-means++')
            model.fit(data_Tsne)
            pred_y = model.predict(data_Tsne)
            C = silhouette_score(data_Tsne, pred_y, sample_size=len(data_Tsne), metric='euclidean')
            fit.append(C)
        # print(fit)
        print(fit)
        fit_arry=np.array(fit)
        return fit_arry

    def evolve(self):
        global x
        # fig = plt.figure()
        global_x=[]
        for step in range(self.max_steps):
            print(step)
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            # 更新速度和权重
            self.v = self.w * self.v + self.c1 * r1 * (self.p - x) + self.c2 * r2 * (self.pg - x)
            x = self.v + x
            fitness = self.calculate_fitness(x)
            # 需要更新的个体
            print(fitness)
            update_id = np.less(self.individual_best_fitness, fitness)         ###????????????
            print(update_id)
            self.p[update_id] = x[update_id]
            self.individual_best_fitness[update_id] = fitness[update_id]
            # 新一代出现了更小的fitness，所以更新全局最优fitness和位置
            if np.max(fitness) > self.global_best_fitness:
                self.pg = x[np.argmax(fitness)]               ####################这里改成最大
                self.global_best_fitness = np.max(fitness)
                global_x=x
            print('best fitness: %.5f, mean fitness: %.5f' % (self.global_best_fitness, np.mean(fitness)))
        print(self.pg.tolist())

x=np.array([
[0.0681, 0.1865, 0.0896, 0.136, 0.1536, 0.023, 0.0895, 0.0238, 0.0238, 0.0799, 0.0721, 0.0336, 0.0205],
[0.121, 0.102, 0.089, 0.088, 0.087, 0.082, 0.077, 0.072, 0.071, 0.069, 0.057, 0.05, 0.034]
    ])
pso = PSO(2, 150)
pso.evolve()
