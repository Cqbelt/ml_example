from sklearn import datasets
from sklearn.model_selection import train_test_split,cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#载入数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.2,random_state=3)

#搜索最佳K值
cv_scores =[]
for n in range(1,31):
    knn = KNeighborsClassifier(n)
    scores = cross_val_score(knn,train_X,train_y,cv=10,scoring='accuracy')
    cv_scores.append(scores.mean())

plt.plot(range(1,31),cv_scores)
plt.xlabel('K')
plt.ylabel('scores')
plt.show()
best_knn = KNeighborsClassifier(n_neighbors=3)
best_knn.fit(train_X,train_y)
print(best_knn.score(test_X,test_y))
