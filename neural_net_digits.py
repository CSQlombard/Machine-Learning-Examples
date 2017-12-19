from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def run_network(N):

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
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

    output_pred = np.asarray(clf.predict(X_test))
    coefs = np.asarray(clf.coefs_)

    plt.plot(range(N), accu_train, label='Training_Accuracy')
    plt.plot(range(N), accu_test, label='Test_Accuracy')
    plt.show()

    return output_pred, y_test, coefs

def plot_coefs(coefs,d):
    fig,axes = plt.subplots(5,4)
    vmin, vmax = coefs[d].min(), coefs[d].max()
    for coefs[d], ax in zip(coefs[d].T, axes.ravel()):
        ax.matshow(coefs[d].reshape(8,8), cmap = plt.cm.gray, vmin =.5*vmin, vmax=.5*vmax)
        ax.set_xticks(())
        ax.set_yticks(())

    plt.show()
