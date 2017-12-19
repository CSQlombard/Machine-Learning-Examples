from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris.data
y = iris.target

def run_network(X,y,N):
    accu_train = np.empty(N)
    accu_test = np.empty(N)
    for hn in range(N):
        # Training and Test data
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state = 21, stratify=y)

        clf = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(hn+1), random_state=1)
        clf.fit(X_train,y_train)
        ## Prediction
        accu_train[hn] = clf.score(X_train,y_train)
        accu_test[hn] = clf.score(X_test,y_test)

    output_pred = clf.predict(X_test)

    plt.plot(range(N), accu_train, label='Training_Accuracy')
    plt.plot(range(N), accu_test, label='Test_Accuracy')
    plt.show()

    return output_pred, y_test
