import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx  # 1
import math
from numpy import array
import sklearn.cluster as sc
import sklearn.metrics as sm
from sklearn.metrics import silhouette_score, davies_bouldin_score

np.seterr(divide='ignore',invalid='ignore')
###1000dataPSO
# weight=[0.10229167795697301, 0.13480688930986873, 0.06431353481586892, 0.09337908112076884, 0.46566515165488326, 0.019248146799812137, 0.09621457029037868, 0.005993617374270608, 0.008632355835419984, -0.01368120869291952, 0.006762796305932098, 0.01060732744209362, 0.00576605978664961]
###2000dataPSO
weight=[0.1241349838691157, 0.1196645100540672, 0.08951770161774118, 0.1636828225167978, 0.1567565935867931, 0.02282540465495746, 0.11819110294401028, -0.003971870861519672, 0.025473215172391206, 0.06157665277750779, 0.0829244054601462, 0.024013494753288445, 0.015210983454703408]
def cal_pairwise_dist1(x):
    #calculate pairwise, x is matrix,(a-b)^2 = a^2 + b^2 - 2*a*b
    sum_x = np.sum(np.square(x), 1)
    # [[1,1],
    #  [2,2]] ,sum_x=[2,8]
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    #(a-b)^2 = a^2 + b^2 - 2*a*b
    return dist

def cal_pairwise_dist2(x):
    y=np.square(x)*weight
    sum_x = np.sum(y, 1)
    # [[1,1],
    #  [2,2]] ,sum_x=[2,8]
    dist = np.add(np.add(-2 * np.dot(x*weight, x.T), sum_x).T, sum_x)
    #(a-b)^2 = a^2 + b^2 - 2*a*b
    return dist

def cal_perplexity(dist, idx=0, beta=1.0):
    prob = np.exp(-dist * beta)
    # 设置自身prob为0
    prob[idx] = 0
    sum_prob = np.sum(prob)
    perp = np.log(sum_prob) + beta * np.sum(dist * prob) / sum_prob #perplexity
    prob /= sum_prob #Pi|j
    return perp, prob


def seach_prob(x, tol=1e-5, perplexity=30.0):

    print("Computing pairwise distances...")
    (n, d) = x.shape
    dist = cal_pairwise_dist2(x)
    pair_prob = np.zeros((n, n))
    beta = np.ones((n, 1))
    base_perp = np.log(perplexity)

    #所有的点
    for i in range(n):
        # if i % 500 == 0:
        #     print("Computing pair_prob for point %s of %s ..." %(i,n))

        betamin = -np.inf
        betamax = np.inf
        perp, this_prob = cal_perplexity(dist[i], i, beta[i])

        perp_diff = perp - base_perp
        tries = 0
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

            perp, this_prob = cal_perplexity(dist[i], i, beta[i])
            perp_diff = perp - base_perp
            tries = tries + 1
        pair_prob[i,] = this_prob
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))#开方，求平均
    return pair_prob

def tsne(x, no_dims=2, initial_dims=50, perplexity=30.0, max_iter=1000):
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

    P = seach_prob(x, 1e-5, perplexity)
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

if __name__ == "__main__":
    data=pd.read_csv("handled_medical record_data(1000).csv")
    data=np.array(data)
    # data = data[:2000, :]
    X = np.array(data)
    Y = tsne(X, 2, 50, 40.0)
    model = sc.KMeans(n_clusters=18)
    model.fit(Y)
    pred_y = model.predict(Y)
    score1 = silhouette_score(Y, pred_y, sample_size=len(Y), metric='euclidean')
    score2 = davies_bouldin_score(Y, pred_y)
    print('Tsne')
    print('silhouette_score(轮廓系数):', score1)
    print('davies_bouldin_score(DBI):', score2)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # plt.title('改进t-SNE')
    plt.scatter(Y[:, 0], Y[:, 1], c='black')
    plt.show()
