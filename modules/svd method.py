from numpy import linalg as la
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('mnist_data49.csv',engine='python')
data=data.drop(columns=["class","Unnamed: 0"])

print(data)
X=np.array(data)
print(X.shape)
U,sigma,VT=la.svd(X)
data=sigma
data1=[]
sum_all=sum(list(map(float,data)))
for i in list(map(float,data)):
    data1.append(i/sum_all)
print(data1)
