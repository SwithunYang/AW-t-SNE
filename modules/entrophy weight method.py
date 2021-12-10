# !/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from numpy import array

df = pd.read_csv('mnist_data49.csv')
df=df.drop(columns=["class","Unnamed: 0"])
print(df)
df.dropna()


def cal_weight(x):
    x = x.apply(lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))))

    rows = x.index.size
    cols = x.columns.size
    k = 1.0 / math.log(rows)

    lnf = [[None] * cols for i in range(rows)]
    # p=array(p)
    x = array(x)
    lnf = [[None] * cols for i in range(rows)]
    lnf = array(lnf)
    for i in range(0, rows):
        for j in range(0, cols):
            if x[i][j] == 0:
                lnfij = 0.0
            else:
                p = x[i][j] / x.sum(axis=0)[j]
                lnfij = math.log(p) * p * (-k)
            lnf[i][j] = lnfij
    lnf = pd.DataFrame(lnf)
    E = lnf

    d = 1 - E.sum(axis=0)
    w = [[None] * 1 for i in range(cols)]
    for j in range(0, cols):
        wj = d[j] / sum(d)
        w[j] = wj

    w = pd.DataFrame(w)
    return w


if __name__ == '__main__':
    w = cal_weight(df)
    w.index = df.columns
    w.columns = ['weight']
    print(np.array(w).ravel().tolist())