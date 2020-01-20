# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:47:36 2019

@author: thyag
"""

import numpy as np

X = np.array([[1,1],[2,3]])
y = np.array([1,-1])
w = np.array([0,1,1])
alpha = 1

def activate_function_theta(x_resul):
    if x_resul > 0:
        return 1
    elif x_resul < 0:
        return -1
    return 0
    
def perceptron(w,x):
    n = len(x)
    resul = []    
    ones = np.ones((n,1))
    x = np.hstack((ones,x))
    for index in range(len(x)):
        resul.append(w.T.dot(x[index]))
    return resul
 
def perceptron_rule(x_example,w,y_hat_example,y_example,alpha):
    x_example = np.append([1],x_example)
    for i in range(len(w)):
        w[i] += alpha*(y_example - y_hat_example)*x_example[i]
    return w


def learning_weights(X,y,w,alpha,epoch):
    for i in range(epoch):
        cont = 0
        resul = list(map(activate_function_theta,perceptron(w,X)))
        for j in range(len(resul)):
            if y[j] != resul[j]:
                w = perceptron_rule(X[j],w,resul[j],y[j],alpha)
                cont += 1
        if (cont == 0):
            break
    return w

w_correct = learning_weights(X,y,w,alpha,20)
y_hat = list(map(activate_function_theta,perceptron(w_correct,X)))
print("Pesos corretos : " + str(w_correct))
print("Checando...")
for i in range(len(y)):
    print("Previsto : {}, Correto : {}".format(y[i],y_hat[i]))








