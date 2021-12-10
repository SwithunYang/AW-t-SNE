from sklearn.manifold import TSNE
import numpy as np
import pylab
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score,precision_score,f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB

data_zip=pd.read_csv('mnist_data49.csv')
n,m=data_zip.shape
label=data_zip.iloc[:,m-1:]
data=data_zip.iloc[:,1:m-1]
X = np.array(data)
print(X)
label=np.array(label)
#Binarization process
for i in range(len(X)):
    for j in range(len(X[i])):
        if X[i][j]!=0:
            X[i][j]=int(1)
        else:
            X[i][j] = int(0)
tsne = TSNE(n_components=2, init='pca', random_state=0,perplexity=30,early_exaggeration=4,learning_rate=100,n_iter=500)
data = tsne.fit_transform(X)
data4=[]
data9=[]
for i in range(0,len(label)):
    if label[i]==4:
        data4.append(data[i])
    if label[i]==9:
        data9.append(data[i])
data4 = np.array(data4)
data9 = np.array(data9)
fig=plt.figure(1)
ax=plt.subplot(111)
type1=ax.scatter(data4[:,0],data4[:,1],c='yellow')
type2=ax.scatter(data9[:, 0], data9[:, 1], c='brown')
ax.legend((type1,type2),('4','9'))
# plt.title("t-SNE")
plt.show()

###Training set and test set splitting
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.9, random_state=2)
print(X_train.shape, X_test.shape)

y_train=y_train.ravel()
y_test=y_test.ravel()

###Classification algorithm model construction
gnb = GaussianNB().fit(X_train, y_train)
y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(cm)
# print(y_test)
# print(y_pred)
print(recall_score(y_true=y_test, y_pred=y_pred, average=None))
print(precision_score(y_true=y_test, y_pred=y_pred, average=None))
# print(f1_score(y_true=y_test, y_pred=y_pred, average='macro'))