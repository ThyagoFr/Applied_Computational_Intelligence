import numpy as np
import gradient_


data = np.loadtxt("files/ex1data1.txt",delimiter = ",")
y = data[:,1]
x = np.column_stack((np.ones((len(y),1)),data[:,0]))
theta = np.array([0,0])

print("RETORNO")
print(gradient_.gradient_desc(x,y,theta,0.01,1000))