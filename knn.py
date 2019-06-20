# dependences
import sys
sys.path.append("plots")
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# load data
dataset_iris = load_iris()

data = dataset_iris.data
feature_names = dataset_iris.feature_names
target = dataset_iris.target
target_names = dataset_iris.target_names

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=4)

k_range = range(1, 100, 2)
scores = {}
scores_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(data_train, target_train)
    target_predict = knn.predict(data_test)
    accuracy = metrics.accuracy_score(target_test, target_predict)
    scores[k] = accuracy
    scores_list.append(accuracy)

plt.plot(k_range, scores_list)
plt.xlabel("Value of k for NN")
plt.ylabel("Testing Accuracy")
plt.grid()
plt.savefig("plots/knn.eps")