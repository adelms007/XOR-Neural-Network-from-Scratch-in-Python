import math
import numpy as np
def sigmoid(z):
    e = math.exp(-z)
    return 1/(1+e)


x_train = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],  
    [1, 1, 1] 
])
y_train = np.array([0, 1, 1, 0])

np.random.seed(10)

neuron_1 = np.random.uniform(-1, 1, 3)
neuron_2 = np.random.uniform(-1, 1, 3)
neuron_3 = np.random.uniform(-1, 1, 3)

lear_rate = 0.5
neurons=[neuron_1,neuron_2,neuron_3]

lear_rate=0.7

neuron_output=np.array([0.0,0.0,0.0])

err=np.array([0.0,0.0,0.0])

def forward(i):
    neuron_output[0]=sigmoid(np.dot(x_train[i],neuron_1))
    neuron_output[1]=sigmoid(np.dot(x_train[i],neuron_2))
    x=np.dot(neuron_3[1:3],neuron_output[0:2])
    neuron_output[2]=sigmoid(neuron_3[0]+x)

def back(y):
    global neuron_1, neuron_2, neuron_3
    der_cost=neuron_output[2]-y
    err[2]=der_cost*neuron_output[2]*(1-neuron_output[2])
    err[1]=err[2]*neuron_3[2]*neuron_output[1]*(1-neuron_output[1])
    err[0]=err[2]*neuron_3[1]*neuron_output[0]*(1-neuron_output[0])
def update(i):
    #neuron 3
    neuron_3[0]-=lear_rate*err[2]*1
    neuron_3[1]-=lear_rate*err[2]*neuron_output[0]
    neuron_3[2]-=lear_rate*err[2]*neuron_output[1]
    #neuron 2
    neuron_2[0]-=lear_rate*err[1]*x_train[i][0]
    neuron_2[1]-=lear_rate*err[1]*x_train[i][1]
    neuron_2[2]-=lear_rate*err[1]*x_train[i][2]
    #neuron 1
    neuron_1[0] -= lear_rate * err[0] * x_train[i][0]
    neuron_1[1] -= lear_rate * err[0] * x_train[i][1]
    neuron_1[2] -= lear_rate * err[0] * x_train[i][2]



for epoch in range(10000):
    for i in range(len(x_train)):
        forward(i)           # 1. On prédit
        back(y_train[i])     # 2. On calcule les erreurs
        update(i)            # 3. On met à jour les poids

for i in range(len(x_train)):
    forward(i)
    print(f"Entrées: {x_train[i][1:]} | Cible: {y_train[i]} | Prédiction: {neuron_output[2]:.4f}")



