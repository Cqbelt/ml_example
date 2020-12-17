from sklearn.datasets import make_blobs 
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(24)
# 产生数据集
X, y = make_blobs(n_samples=1000, n_features=2, centers=4, random_state=8)  # X是数据特征值集合，y是类别
 
fig, ax1 = plt.subplots(1)  # 创建一个子图,返回 Figure对象（fig) 和 子图对象（ax1）
ax1.scatter(X[:, 0], X[:, 1], marker='o', s=8) 
plt.show()

# 不同簇显示不同的颜色，这里是数据实际的分类
color = ["red","pink","orange","gray"] 
fig, ax1 = plt.subplots(1)
for i in range(4):
    ax1.scatter(X[y==i, 0], X[y==i, 1], marker='o', s=8, c=color[i])
plt.xlabel('X')
plt.ylabel('y')
plt.show()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples 
K = [3,4,5,6]  # 猜测簇的个数
# KMeans对象有三个属性：labels_,cluster_centers_,inertia_
# 属性 labels_：模型 聚类 得到的类别（每个样本所对应类）
# 属性 cluster_centers_：最终得到的所有的质心
# 属性 inertia_：总距离平方和---越小越好
scores = []
for k in K:
    cl = KMeans(n_clusters=k,random_state=24).fit(X)
    print(cl.inertia_)
    score = silhouette_score(X, cl.labels_)
    # 使用k=3时，数据集 得到的聚类效果（轮廓系数）
    scores.append(score)
#实际上，当我们设的 k 越大（不超过样本数），总距离平方和就会越小。这是因为，k越大，则质心就越多，例如当k为500时，那么就相当于每一个数据点都是一个质心，那此时总距离平方和直接就会等于0了

print()
print(scores)

