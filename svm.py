# dependences
import sys
sys.path.append("plots")
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn import svm
# load data
dataset_iris = load_iris()

data = dataset_iris.data
feature_names = dataset_iris.feature_names
target = dataset_iris.target
target_names = dataset_iris.target_names

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=4)

c_range = range(1, 1000)
scores = {}
scores_list = []
for c in c_range:
    clf = svm.SVC(gamma=0.01, C=c, kernel='rbf')
    clf.fit(data_train, target_train)
    target_predict = clf.predict(data_test)
    accuracy = metrics.accuracy_score(target_test, target_predict)
    scores[c] = accuracy
    scores_list.append(accuracy)

plt.plot(c_range, scores_list)
plt.xlabel("Value of C for SVM")
plt.ylabel("Testing Accuracy")
plt.savefig("plots/svm.eps")