from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(24)
# 产生数据集
X, y = make_blobs(n_samples=200, n_features=3, centers=4, random_state=8)  # X是数据特征值集合，y是类别
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)

clf = GaussianNB()
clf.fit(X_train, y_train)
plt.scatter(X[X[:,0]<4][:,0],X[X[:,0]<4][:,1])
plt.scatter(X[X[:,0]>4][:,0],X[X[:,0]>4][:,1])
plt.xlabel('X')
plt.ylabel('y')

y_pred=clf.predict(X_test)
score=clf.score(X_test, y_test)
print(score)
plt.plot([2,6],[10,-7.5],'r-')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
