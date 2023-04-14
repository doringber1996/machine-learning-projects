# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:08:39 2023

@author: Dor Ingber 316080159
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('sample.csv')


X = dataset.iloc[:, 0]
Y = dataset.iloc[:, 1]
# plt.scatter(X,Y)
# plt.show()

#Initialise parameters
a = 0.0
b = 0.0
l = 0.0001
L = 0.1
A = []
B = []
LOSS = []
Batch_Size=32

def Batch_Gradient_Descent(X,Y,a,b,alpha,iteration=1000):
    
    N = float(len(X))
    erorlst = []
    for epoch in range(iteration):
        yhat = a*X +b
        grad_a = (-2/N)*sum(X*(Y-yhat))
        grad_b = (-2/N)*sum(Y-yhat)
        a = a- alpha*grad_a
        b = b- alpha*grad_b
        lst = []
        for k in (Y-yhat):
            lst.append(k**2)
        loss = (1/N)*sum(lst)
        A.append(a)
        B.append(b)
        LOSS.append(loss)
        I = epoch+1
        erorlst.append(I)
    return A,B,LOSS,erorlst

def Stochastic_Gradient_Descent(X,Y,a,b,alpha,iteration=1000):
    
    N = float(len(X))
    erorlst = []
    for epoch in range(iteration):
        r=np.random.randint(N-1)
        yhat = a*X[r] +b
        grad_a = (-2/1)* X[r]*(Y[r]-yhat)
        grad_b = (-2/1)*(Y[r]-yhat)
        a = a - alpha*grad_a
        b = b - alpha*grad_b
        loss = (1/N)*((Y[r]-yhat)**2)
        A.append(a)
        B.append(b)
        LOSS.append(loss)
        I = epoch+1
        erorlst.append(I)
    return A,B,LOSS,erorlst

def Mini_Batch_Gradient_Descent(X,Y,a,b,alpha,iteration=1000,BatchSize=32):
    
    N = float(len(X))
    erorlst = []
    for epoch in range(iteration):
        r =np.random.choice(range(0, int(N-1)), BatchSize)
        for xi,yi in zip(X[r],Y[r]):
            yhat = a*xi +b
            grad_a = (-2/BatchSize)* xi*(yi-yhat)
            grad_b = (-2/BatchSize)*(yi-yhat)
            a = a - alpha*grad_a
            b = b - alpha*grad_b
            loss = (np.sum(((yi-yhat)**2),axis=0)/N) #x.shape[0]=N
        A.append(a)
        B.append(b)
        LOSS.append(loss)
        I = epoch+1
        erorlst.append(I)

    return A,B,LOSS,erorlst

#Batch Gradient Descent,L=0.0001

A,B,LOSS,erorlst=Batch_Gradient_Descent(X,Y,a,b,l,1000)

plt.plot(np.array(erorlst),np.array(LOSS))
plt.title('loss of Bach Gradient Descent alpha=0.0001')
plt.xlabel('epoch')
plt.yscale('log')
plt.ylabel('loss')
plt.show()

plt.plot(np.array(erorlst),np.array(A))
plt.title('Slope of Bach Gradient Descent alpha=0.0001')
plt.xlabel('epoch')
plt.ylabel('a')
plt.show()

plt.plot(np.array(erorlst),np.array(B))
plt.title('intersept of Bach Gradient Descent alpha=0.0001')
plt.xlabel('epoch')
plt.ylabel('b')
plt.show()

# Batch Gradient Descent,L=0.1

A,B,LOSS,erorlst=Batch_Gradient_Descent(X,Y,a,b,L,1000)

plt.plot(np.array(erorlst),np.array(LOSS))
plt.title('loss of Bach Gradient Descent alpha=0.1')
plt.xlabel('epoch')
plt.yscale('log')
plt.ylabel('loss')
plt.show()

plt.plot(np.array(erorlst),np.array(A))
plt.xlabel('epoch')
plt.ylabel('a')
plt.show()

plt.plot(np.array(erorlst),np.array(B))
plt.xlabel('epoch')
plt.ylabel('b')
plt.show()

# Stochastic Gradient Descent,L=0.0001

A,B,LOSS,erorlst=Stochastic_Gradient_Descent(X,Y,a,b,l,1000)

plt.plot(np.array(erorlst),np.array(LOSS))
plt.title('loss of Stochastic Gradient  alpha=0.0001')
plt.xlabel('epoch')
plt.yscale('log')
plt.ylabel('loss')
plt.show()

plt.plot(np.array(erorlst),np.array(A))
plt.title('Slope of Stochastic Gradient alpha=0.0001')
plt.xlabel('epoch')
plt.ylabel('a')
plt.show()

plt.plot(np.array(erorlst),np.array(B))
plt.title('intersept of Stochastic Gradient alpha=0.0001')
plt.xlabel('epoch')
plt.ylabel('b')
plt.show()

# Stochastic Gradient Descent,L=0.1

A,B,LOSS,erorlst=Stochastic_Gradient_Descent(X,Y,a,b,L,1000)

plt.plot(np.array(erorlst),np.array(LOSS))
plt.title('loss of Stochastic Gradient  alpha=0.1')
plt.xlabel('epoch')
#plt.yscale('log')
plt.ylabel('loss')
plt.show()

plt.plot(np.array(erorlst),np.array(A))
plt.title('Slope of Stochastic Gradient alpha=0.1')
plt.xlabel('epoch')
plt.ylabel('a')
plt.show()

plt.plot(np.array(erorlst),np.array(B))
plt.title('intersept of Stochastic Gradient alpha=0.1')
plt.xlabel('epoch')
plt.ylabel('b')
plt.show()


# Mini Batch Gradient Descent,L=0.0001

A,B,LOSS,erorlst=Mini_Batch_Gradient_Descent(X,Y,a,b,l,1000,Batch_Size)

plt.plot(np.array(erorlst),np.array(LOSS))
plt.title('loss of Mini Bach Gradient  alpha=0.0001')
plt.xlabel('epoch')
plt.yscale('log')
plt.ylabel('loss')
plt.show()

plt.plot(np.array(erorlst),np.array(A))
plt.title('Slope of Mini Bach Gradient alpha=0.0001')
plt.xlabel('epoch')
plt.ylabel('a')
plt.show()

plt.plot(np.array(erorlst),np.array(B))
plt.title('intersept of Mini Bach Gradient alpha=0.0001')
plt.xlabel('epoch')
plt.ylabel('b')
plt.show()

# Mini Batch Gradient Descent,L=0.1

A,B,LOSS,erorlst=Mini_Batch_Gradient_Descent(X,Y,a,b,L,1000,Batch_Size)

plt.plot(np.array(erorlst),np.array(LOSS))
plt.title('loss of Mini Bach grad  alpha=0.1')
plt.xlabel('epoch')
plt.yscale('log')
plt.ylabel('loss')
plt.show()

plt.plot(np.array(erorlst),np.array(A))
plt.title('Slope of Mini Bach Gradient alpha=0.1')
plt.xlabel('epoch')
plt.ylabel('a')
plt.show()

plt.plot(np.array(erorlst),np.array(B))
plt.title('intersept of Mini Bach Gradient alpha=0.1')
plt.xlabel('epoch')
plt.ylabel('b')
plt.show()

print("parameter a (slope) is:\n",A[-1],"\nparameter b (intersept) is:\n",B[-1],"\nLoss is:\n",LOSS[-1])
