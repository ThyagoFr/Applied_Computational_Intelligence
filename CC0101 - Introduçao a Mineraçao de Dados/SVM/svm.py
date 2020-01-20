# IMD, 2019.2, UFC/DEMA
import numpy as np
from sklearn import datasets, svm
import sklearn.model_selection as msel

#------------------------------------------------------------------------------
# Leitura dos dados do dataset MNIST

dados = datasets.load_digits()
X = dados.data
y = dados.target

[m,n] = dados.data.shape
print('Numero de exemplos:', m)
print('Numero de caraceristicas:', n, end='\n\n')

#------------------------------------------------------------------------------

# Criar classificador
mvs = svm.SVC(gamma='auto')

# Esquema de particao dos dados
particao = msel.RepeatedStratifiedKFold(n_splits=5, n_repeats=5)

# Precisao da validacao cruzada com parametros default
precisoes = msel.cross_val_score(mvs, X, y, cv=particao)
print('Precisao media (default):',
      np.round(np.average(precisoes), 4), end='\n')


# Dicionario com parametros que serao ajustados e seus valores
parametros = {
        'gamma': ['auto'],
        'C': [0.1,1.0, 2.5, 5.0, 10.0],
        'kernel': ['linear', 'poly'],
        'degree': [2, 3, 4]
        }

# Criar objeto responsavel pela busca em grade
gs = msel.GridSearchCV(estimator=mvs,
                       param_grid=parametros,
                       scoring='accuracy',
                       cv=particao)

gs.fit(X,y) # Realiza os testes propriamente ditos buscando os melhores paremtros

print('Melhores parametros:', gs.best_params_)

melhor = svm.SVC(**gs.best_params_)
precisoes = msel.cross_val_score(melhor, X, y, cv=particao)
print('Precisao media (melhores parametros):',
      np.round(np.average(precisoes), 4), end='\n')
