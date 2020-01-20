# UFC/DEMA, Introducao 'a Mineracao de Dados, 2019.2
# Atividade nao-presencial, dia 13 de agosto de 2019
#
# Dado um conjunto D de m pontos em R^n gerados aleatoriamente, encontrar um  ponto "central" deste conjunto.
#
# Neste exercicio, um ponto P é central em D se P possui o menor "raio" dentre
# todos os pontos em D.
#
# O raio de um ponto P em D é simplesmente a maior distancia entre P e algum
# outro ponto de D.
# ----------------------------------------------------------------------------------
# 
# Seu codigo deve imprimir uma mensagem, informando qual é o ponto de menor raio,
# isto é, qual é o ponto central.

import numpy as np
import math as m

np.random.seed(9)
num_pontos = 10
num_coordenadas = 3

def dist_eucli(ponto_a,ponto_b):
  soma = sum(map(lambda x,y : (x-y)**2,ponto_a,ponto_b))
  return m.sqrt(soma)

def calc_raio(array_valores):
  dist,aux = [],[]
  n_pontos = array_valores.shape[0]
  for i in range(n_pontos):
    for j in range(n_pontos):
      aux.append(dist_eucli(array_valores[i],array_valores[j]))
    dist.append(max(aux))
    aux = []
  return (np.argmin(dist),dist[np.argmin(dist)])

Dados = np.random.rand(num_pontos, num_coordenadas)
ponto,valor = calc_raio(Dados)
print("Ponto : {}\nRaio : {}".format(ponto,valor))