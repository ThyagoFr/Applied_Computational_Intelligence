#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 10:47:54 2019

@author: Thyago Freitas da Silva - 392035
"""

# Importando datasets que serão utilizados, assim como o numpy
from sklearn.datasets import load_wine
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
import numpy as np

""" 
Funcao responsavel por buscar h e m, dada a equação vista nas notas de aula
Parametros
x = amostras(apenas as features) 
y = classes de cada amostra
class_random_instance = classe da amostra selecionada "aleatoriamente"
random_instance = indice da amostra selecionada "aleatoriamente"
"""
def search_h_m(x,y,class_random_instance,random_instance):
    # Buscando todas os indices que possuam classes DIFERENTES da amostra aleatoria selecionada
    other_classes = np.where(y != class_random_instance)[0]
    
    # Buscando todas os indices que possuam classes DIFERENTES da amostra aleatoria selecionada
    # ** Nesse caso,o indice da amostra aleatoria tambem é retornado,assim
    # ** na hora de buscar h,basta "pular" o primeira indice,pois ele sempre será
    # ** igual ao indice da amostra aleatoria,logo deve-se pegar o segundo indice.
    same_class = np.where(y == class_random_instance)[0]
    
    # Criando um vetor que armazena a tupla (indice,distancia) para todas as amostras
    # com a mesma classe que a amostra aleatoria
    h_distances = [ (i,np.linalg.norm(x[random_instance] - x[i])) for i in same_class]
    
    # Criando um vetor que armazena a tupla (indice,distancia) para todas as amostras
    # com classes diferentes que a amostra aleatoria
    m_distances = [ (i,np.linalg.norm(x[random_instance] - x[i])) for i in other_classes]
    
    # Pegando o indice da amostra que teve a segunda** menor distancia dentre as amostras com a mesma
    # classe.
    h = sorted(h_distances,key=lambda x : x[1])[1][0]
    
    # Pegando o indice da amostra que teve a menor distancia dentre as amostras classes diferentes
    # da amostra aleatoria selecionada
    m = sorted(m_distances,key=lambda x : x[1])[0][0]
    
    # Retorna h e m
    return (h,m)


"""
Implementaçao por função do algoritmo Relief
x = amostras(apenas as features) 
y = classes de cada amostra
t = Numero de iterações
"""
def relief(x,y,t = 100):
    
    # Numero de amostras
    n_instances = len(y)
    
    # Inicializando todos os pesos das amostras com 0
    weights = np.zeros(x.shape[1])
    
    # Loop de 100 vezes(default)
    for k in range(t):
        
        #Seleciona "aleatoriamente" o indice de uma amostra
        random_instance = np.random.randint(n_instances)
        
        #Salva a classe dessa amostra
        class_random_instance = y[random_instance]
        
        #Repassa os dados para a comentada anteriormente
        h,m = search_h_m(x,y,class_random_instance,random_instance)
        
        # Calculo dos pesos de todas as features de uma vez utilizando a equação das notas de aula
        weights = weights - (x[random_instance] - x[h])**2 + (x[random_instance] - x[m])**2
    
    # Criando labels para as features (apenas para apresentaçao)
    features = ["Feature {}".format(i) for i in range(n_instances)]
    
    # Associando os pesos as suas labels (apenas para apresentaçao)
    features_weights = list(zip(features, np.around(weights,decimals=4)))
    
    # Ordenando de forma decrescente o rank (caracteristica mais importante para menos importante)
    rank = sorted(features_weights,key=lambda x: x[1],reverse=True)
    return rank

# Criando um vetor vazio para armazenar os objetos contendo os datasets e suas informaçoes
data = []
data.append(load_digits())
data.append(load_iris())
data.append(load_breast_cancer())
data.append(load_wine())

# Percorrendo o vetor
for k in data:
    # Repassando para a funcao relief os dados de cada objeto, k.data = x e k.target = y
    new_weights = relief(k.data,k.target)
    
    # Printando o nome do dataset
    print(k.DESCR.split("\n")[2])
    
    # Printando o rank retornado pelo algoritmo Relief
    for i in new_weights:
        print("{}, \t Importancia = {}".format(*i))
    print("\n")
        