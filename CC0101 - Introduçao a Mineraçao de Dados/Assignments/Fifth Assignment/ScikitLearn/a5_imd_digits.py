from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

import numpy as np
from sklearn.datasets import load_digits

import warnings
warnings.filterwarnings('ignore') 


"""### Carregando dataset e criando partição estratificada"""
data = load_digits()
x = data.data
y = data.target

partition = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)

"""### KNN - Busca em grade"""

#help(KNeighborsClassifier)
knn = KNeighborsClassifier()

knn_grid_params = {
    'n_neighbors' : [3,5,7],
    'weights' : ['uniform','distance'],
}

gs_knn = GridSearchCV(estimator=knn,param_grid=knn_grid_params,scoring='accuracy',cv=partition)
gs_knn.fit(x,y)

"""#### Melhores parametros encontrados"""

print("Utilizando KNN...")
print("Melhores parametros : " + str(gs_knn.best_params_))

"""#### Acuracia média usando cross validation"""

best_knn = KNeighborsClassifier(**gs_knn.best_params_)
accuracys = cross_val_score(best_knn, x, y, cv = partition)
print('Acuracia media (melhores parametros): {}%'.format(np.round(100*np.average(accuracys))))

"""### Random Forest - Busca em grade"""

# help(RandomForestClassifier)
rf = RandomForestClassifier()
rf_grid_params = {
    'n_estimators' : [25,50],
    'criterion' : ['gini','entropy'],
    'max_depth' : [10,15]
}

gs_rf = GridSearchCV(estimator=rf,param_grid=rf_grid_params,scoring='accuracy',cv=partition)
gs_rf.fit(x,y)

"""#### Melhores parametros encontrados"""

print("Utilizando Random Forest...")
print("Melhores parametros : " + str(gs_rf.best_params_))

"""#### Acuracia média usando cross validation"""

best_rf = RandomForestClassifier(**gs_rf.best_params_)
accuracys = cross_val_score(best_rf,x,y,cv=partition)
print('Acuracia media (melhores parametros): {}%'.format(np.round(100*np.average(accuracys))))

"""### MultiLayer Perceptron - Busca em grade"""

# help(MLPClassifier)
mlp = MLPClassifier()
mlp_grid_params = {
    'activation' : ['logistic','relu'],
    'max_iter' : [500,600],
    'hidden_layer_sizes' : [100,120]
}

gs_mpl = GridSearchCV(estimator=mlp,param_grid=mlp_grid_params,scoring="accuracy",cv=partition)
gs_mpl.fit(x,y)

"""#### Melhores parametros encontrados"""
print("Utilizando MLP...")
print("Melhores parametros : " + str(gs_mpl.best_params_))

"""#### Acuracia média usando cross validation"""

best_mpl = MLPClassifier(**gs_mpl.best_params_)
accuracys = cross_val_score(best_mpl,x,y,cv=partition)
print("Acuracia media (melhores parametros) : {}%".format(np.round(100*np.average(accuracys))))

"""### Support Vector Machine - Busca em grade"""

#help(SVC)
svm = SVC(gamma="scale")
svm_grid_params = {
    'C' : [1,0.5,2],
    'kernel':['linear','poly','sigmoid']
}

gs_svm = GridSearchCV(estimator=svm,param_grid=svm_grid_params,scoring="accuracy",cv=partition)
gs_svm.fit(x,y)

"""#### Melhores parametros encontrados"""
print("Utilizando SVM...")
print("Melhores parametros : " + str(gs_svm.best_params_))

"""#### Acuracia média usando cross validation"""

best_svm = SVC(**gs_svm.best_params_)
accuraccys = cross_val_score(best_svm,x,y,cv=partition)
print("Acuracia media (melhores parametros) : {}%".format(np.round(100*np.average(accuracys))))
