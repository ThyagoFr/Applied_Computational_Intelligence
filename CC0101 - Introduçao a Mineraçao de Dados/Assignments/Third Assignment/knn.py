# Introducao aa Mineracao de Dados, 2019.2, UFC/DEMA
# Implementacao basica do classificador de k vizinhos mais proximos (k-NN)

import numpy as np
import matplotlib.pyplot as plt

def distanciaEuclidiana(a, b):
    d = (a[0] - b[0])**2 + (a[1] - b[1])**2
    return np.sqrt(d)

#------------------------------------------------------------------------------
# Dimensoes dos dados

m = 100 # Numero de exemplos na amostra
k = 2   # Numero de vizinhos
t = 1000 # Numero de exemplos de teste

# Gerar exemplos de treinamento aleatoriamente

np.random.seed(88)    # Semente do gerador aleatorio
S = np.random.rand(m,2) # Exemplos da amostra
S = [list(x) for x in S] # Transforma amostra em lista de listas

# Informacao de classe
classe = [1]*m
for i in range(m):
    d = distanciaEuclidiana(S[i], [0.5, 0.5])
    if d < 0.3:
        classe[i] = 0

# Plotar dados de treinamento
for i in range(m):
    if classe[i] == 0:
        plt.scatter(x=[S[i][0]], y=[S[i][1]], c='b', marker='o')
    else:
        plt.scatter(x=[S[i][0]], y=[S[i][1]], c='m', marker='o')

plt.show()

#------------------------------------------------------------------------------
# Gerar dados de teste
Teste = np.random.rand(t,2)
Teste = [list(x) for x in Teste]

#------------------------------------------------------------------------------
# Algoritmo k nearest neighbors (k-NN).

# Definir a distancia a ser utilizada
dist = distanciaEuclidiana

# Para cada exemplo de teste
for i in range(t):

    # Gerar exemplo
    ex = Teste[i]

    # Calcular distancia a todos os exemplos de S
    # A lista <distancias> contem pares ordenados do tipo (d,j), onde j e'
    # o indice do exemplo de treinamento e d e' a distancia do j-esimo exemplo
    # de treinamento para o exemplo de teste
    distancias = [ (dist(ex, S[j]), j) for j in range(m) ]

    # Determinar vizinhos mais proximos:
    # (ordena lista em ordem crescente da distancia)
    distancias.sort()

    # -------------------------------------------------------------------------
    # Obter indices e classes dos exemplos mais proximos
    indices_dos_mais_proximos = [ distancias[i][1] for i in range(k) ]
    classes_dos_mais_proximos = [ classe[i] for i in indices_dos_mais_proximos ]
    cmp = classes_dos_mais_proximos
    
    # Obter frequencia de cada classe existente entre os vizinhos mais proximos
    # Cada item de <frequencias> tem a forma (frequencia, classe)
    frequencias = [ (cmp.count(c), c) for c in classes_dos_mais_proximos ]
    
    # Obter a classe mais frequente (ordena por frequencia, ordem decrescente)
    frequencias.sort(reverse=True)
    previsao = frequencias[0][1] # Pegar classe do primeiro elemento

    # Plotar previsoes
    if previsao == 0:
        plt.scatter(x=[ex[0]], y=[ex[1]], c='b', marker='o')
    else:
        plt.scatter(x=[ex[0]], y=[ex[1]], c='m', marker='o')
plt.title("Previsao")
plt.show()

#------------------------------------------------------------------------------
