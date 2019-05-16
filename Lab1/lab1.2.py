# Factorization-Based Data Modeling --  TP 1: non-negative matrix factorization

import numpy as np
import matplotlib.pylab as plt

# Data generation
I=10
J=20
K=3

Wtrue=np.dot(2,np.random.rand(I,K))
Htrue=np.dot(2,np.random.rand(K,J))

randomMask1=(np.random.rand(I,K) < 0.5)
randomMask2=(np.random.rand(K,J) < 0.5)

Wtrue=np.multiply(Wtrue,randomMask1)
Htrue=np.multiply(Htrue,randomMask2)

dataNoise=1
X=np.dot(Wtrue,Htrue) + np.dot(dataNoise,np.random.rand(I,J))

# visualize the data with the true factor matrices
f, ax = plt.subplots(2,2)
ax[0,1].imshow(Htrue)
ax[1,0].imshow(Wtrue)
ax[1,1].imshow(X)

# Algorithm: Multiplicative Update Rules

# Initialize the factor matrices
Wmur=np.dot(2,np.random.rand(I,K))
Hmur=np.dot(2,np.random.rand(K,J))
MaxIterMur = 100;

# record the objective function values
obj_mur = np.zeros((MaxIterMur,1))  

O = np.ones((I,J))
Xhat = Wmur.dot(Hmur)
eps = 0.01

for i in range(MaxIterMur):
    Wmur = np.multiply(Wmur,np.divide(np.divide(X,Xhat).dot(Hmur.transpose()),O.dot(Hmur.transpose())))
    Hmur = np.multiply(Hmur,np.divide(Wmur.transpose().dot(np.divide(X,Xhat)),Wmur.transpose().dot(O)))
    Xhat = Wmur.dot(Hmur)
    Xhat = Xhat + eps
    tmp = np.multiply(X,(np.log(np.divide(X,Xhat)))) - X + Xhat
    obj_mur[i] = sum(sum(tmp))
    # visualize the iterations
    # f, ax = plt.subplots(2,2)
    # ax[0,1].imshow(Hmur)
    # ax[0,1].set_title('H' + str(i + 1))
    # ax[1,0].imshow(Wmur)
    # ax[1,0].set_title('W' + str(i + 1))
    # ax[1,1].imshow(Xhat)
    # ax[1,1].set_title('X' + str(i + 1))

# draw the objective function
f, ax = plt.subplots(1,1)
plt.plot(obj_mur,'r-o')
plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.title('MUR')
plt.show()

