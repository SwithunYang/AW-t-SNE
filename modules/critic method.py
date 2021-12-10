import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
def critic(X):
    n,m = X.shape
    Z = standard(X)  # standard
    Z =X
    # Scaler = StandardScaler().fit(X)
    # Z = Scaler.transform(X)
    R = np.array(pd.DataFrame(Z).corr())
    # print(R)
    # print(R.shape[0])
    delta = np.zeros(m)
    c = np.zeros(m)
    for j in range(m):
        delta[j] = Z[:,j].std()
        c[j] = R.shape[0] - R[:,j].sum()
    C = delta * c
    w = C/sum(C)
    return w

def standard(X):
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    xmaxmin = xmax-xmin
    n, m = X.shape
    for i in range(n):
        for j in range(m):
            X[i,j] = (X[i,j]-xmin[j])/xmaxmin[j]
    return X

if __name__ == '__main__':
    # data = pd.read_csv('mnist_data1.csv',engine='python')
    # digits = datasets.load_digits(n_class=6)
    # data = digits.data
    data = pd.read_csv('mnist_data49.csv', engine='python')
    data = data.drop(columns=["class", "Unnamed: 0"])
    data = np.array(data)
    X=data
    for i in range(len(X)):
        for j in range(len(X[i])):
            if X[i][j]!=0:
                X[i][j]=int(1)
            else:
                X[i][j] = int(0)
    # _range = np.max(X) - np.min(X)
    # X = (X - np.min(X)) / _range
    print(list(critic(X)))