import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

np.random.seed(24)
#加载数据
iris = load_iris()
X = iris.data
X_norm = StandardScaler().fit_transform(X)
X_norm.mean(axis=0)

# 求特征值和特征向量
ew,ev = np.linalg.eig(np.cov(X_norm.T))
# 特征向量特征值的排序
ew_oreder = np.argsort(ew)[::-1]
ew_sort = ew[ew_oreder]
ev_sort = ev[:, ew_oreder]  # ev的每一列代表一个特征向量
ev_sort.shape # (4,4)

# 我们指定降成2维， 然后取出排序后的特征向量的前两列就是基
K = 2
V = ev_sort[:, :2]  # 4*2

# 最后，我们得到降维后的数据
X_new = X_norm.dot(V)    # shape (150,2)

colors = ['red', 'black', 'orange']

plt.figure()
for i in [0, 1, 2]:
    plt.scatter(X_new[iris.target==i, 0],
                X_new[iris.target==i, 1],
                alpha=.7,
                c=colors[i],
                label=iris.target_names[i]
               )

plt.legend()
plt.title('PCa of IRIS dataset')
plt.xlabel('PC_0')
plt.ylabel('PC_1')
plt.show()

#使用库
from sklearn.decomposition import PCA

# 然后使用
pca = PCA(n_components=2)
X_new = pca.fit_transform(X_norm)

"""查看PCA的一些属性"""
print(pca.explained_variance_)    # 属性可以查看降维后的每个特征向量上所带的信息量大小（可解释性方差的大小）
print(pca.explained_variance_ratio_)  # 查看降维后的每个新特征的信息量占原始数据总信息量的百分比
print(pca.explained_variance_ratio_.sum())    # 降维后信息保留量

'''
[4.22824171 0.24267075]   # 可以发现，降维后特征的方差
[0.92461872 0.05306648]  # 降维后的特征带的原有数据的信息量的比例
0.977685206318795      # 降维后的信息保留（损失了3%， 去掉了一半特征，还算可以）
'''

pca_line = PCA().fit(X_norm)
plt.plot([1,2,3,4],np.cumsum(pca_line.explained_variance_ratio_))
plt.xlabel('component')
plt.ylabel('proportion')
plt.xticks([1,2,3,4])
plt.show()

