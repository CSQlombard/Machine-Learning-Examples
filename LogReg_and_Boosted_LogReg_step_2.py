from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.base import TransformerMixin
from scipy.sparse import linalg
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import io
from time import time

# Get the labels for all the data
def get_labels(lista):
    y = np.empty((len(lista),),dtype=int)
    file = io.open('train.csv','r',encoding='utf-8')
    c = 0
    for index, info in enumerate(file.readlines()):
        if index > 0 and index <= len(lista): ## first line are labels
            info = info.split('","')
            if info[2] == u'EAP"\n':
                y[index-1] = int(0)
                c = c + 1
            if info[2] == u'HPL"\n':
                y[index-1] = int(1)
                c = c + 1
            if info[2] == u'MWS"\n':
                y[index-1] = int(2)
                c = c + 1

    if c != len(lista):
        print("Mistake!!")

    return y

## Stratified split
def Strat_Split(lista,y):
    lista_train = []
    lista_test = []
    y_train = []
    y_test = []
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_index, test_index in split.split(lista, y):
        if set(train_index).intersection(test_index):
            print "Overlapping indeces!! Change random state and try again."
        for i,train in enumerate(train_index):
            lista_train.append(lista[train])
            y_train.append(y[train])
        for i,test in enumerate(test_index):
            lista_test.append(lista[test])
            y_test.append(y[test])
        y_train = np.array(y_train)
        y_test = np.array(y_test)
    return lista_train, y_train, lista_test, y_test

def dim_red(X,K):
    U,s,V = linalg.svds(X,K)
    # Rearange output
    n = len(s)
    U[:,:n] = U[:,n-1::-1]
    s = s[::-1]
    x,y = X.shape
    S = np.zeros((K,K), dtype=float)
    S[:K, :K] = np.diag(s[0:K])
    concept_matrix = np.dot(U,S)
    concept_matrix = concept_matrix[:][:,0:K]
    return concept_matrix

# My TfidfTransformer
class Dim_reduction(TransformerMixin):
    def __init__(self, K, n, apply=True, local=True):
        self.apply = apply
        self.K = K
        self.local = local
        self.n = n
    def fit(self, X,y=None):
        return self
    def transform(self,X):
        concept_matrix = []
        if self.apply:
            if self.local:
                lle = LocallyLinearEmbedding(n_components=self.K, n_jobs=-1, n_neighbors=self.n)
                concept_matrix = lle.fit_transform(X.toarray())
            else:
                concept_matrix = dim_red(X,self.K)
        else:
            concept_matrix = X
        return concept_matrix

# dont forget to import all the modules including Dim_reduction
from mejorando import Dim_reduction
my_pipeline = Pipeline([
        ('dict_vect',DictVectorizer(sparse=True)),
        ('transf',TfidfTransformer()),
        ('dim_red',Dim_reduction(100,10,apply=False, local=False)) # local=True does not accept sparse matrices
#        ('max_abs_scaler', MaxAbsScaler()),
#        ('std_scaler', StandardScaler())
])

# Multinomiale Logistische Regression mit GitterSearch
#param_grid = [{'C':range(1,1000,50)}]
def LogReg(X_train,y_train,param_grid):
    start = time()
    softmax_reg=LogisticRegression(n_jobs=-1,multi_class='ovr',solver='lbfgs')
    grid_search = GridSearchCV(softmax_reg, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train,y_train)
    cvres = grid_search.cv_results_
    for score, params in zip(cvres['mean_test_score'],cvres['params']):
        print score, params
    print time()-start

# Boosting apply to logistic regression
# The output compares the logistic regression with the boosted one
param_grid = {'C': [1000, 100000], 'n_estimators': [1000, 2000, 5000], 'learning_rate':[0.01]}
def Log_Boosting(X_train,y_train,X_test, y_test,param_grid):
    c = 0
    for item in ParameterGrid(param_grid):
        start = time()
        if item['C'] != c:
            log_clf = LogisticRegression(C=item['C'],n_jobs=-1,multi_class='ovr',solver='lbfgs')
            log_clf.fit(X_train,y_train)
            accu_train_log = log_clf.score(X_train, y_train)
            accu_test_log = log_clf.score(X_test, y_test)
            c = item['C']
            #print 'logistic'

        ada_clf = AdaBoostClassifier(log_clf, n_estimators = item['n_estimators'],
        algorithm='SAMME.R', learning_rate=item['learning_rate'])
        ada_clf.fit(X_train,y_train)
        accu_train = ada_clf.score(X_train, y_train)
        accu_test = ada_clf.score(X_test, y_test)

        print accu_train, accu_test, accu_train_log, accu_test_log,item,time()-start

"""
# Instructions to run the functions for the classification
After using Text_Classification_step_1.py you will have a file
called lista.
# From python shell do
import LogReg_and_Boosted_LogReg_step_2 as l
# and follow the next steps

1) Get the labels
y = l.get_labels(lista)

2) Apply a Stratified Split to get train and test data
lista_train, y_train, lista_test, y_test = l.Strat_Split(lista,y)

3) Apply the pipeline
X_train = l.my_pipeline.fit_transform(lista_train)
X_test = l.my_pipeline.transform(lista_test)

4) Apply the Boosting classifier (it includes the normal LogReg without boosting)
param_grid = {'C': [100, 1000, 10000], 'n_estimators': [100, 1000], 'learning_rate':[0.001, 0.01]}
l.Log_Boosting(X_train,y_train,X_test, y_test,param_grid)

This will output the training and test accuracy for the boosting (first two outputs)
follow by the training and accuacy for the log Reg (without boosting)
"""
