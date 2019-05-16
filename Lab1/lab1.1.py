# Factorization-Based Data Modeling --  TP 1: matrix factorization

import numpy as np
import matplotlib.pylab as plt

# Data generation
I=10
J=20
K=3

# initialize it from random numbers ("randn" can be positive or negative)
Wtrue=np.dot(2,np.random.randn(I,K))
Htrue=np.dot(2,np.random.randn(K,J))


# make the true factor matrices more sparse
randomMask1=(np.random.rand(I,K) < 0.5)
randomMask2=(np.random.rand(K,J) < 0.5)


Wtrue=np.multiply(Wtrue,randomMask1)
Htrue=np.multiply(Htrue,randomMask2)

# generate the data (multiply the matrices and add some more noise)
dataNoise=1
X=np.dot(Wtrue,Htrue) + np.dot(dataNoise,np.random.randn(I,J))

# visualize the data with the true factor matrices
# method1
plt.subplot(222)
plt.imshow(Htrue)
plt.subplot(223)
plt.imshow(Wtrue)
plt.subplot(224)
plt.imshow(X)


# method2
f, ax = plt.subplots(2,2)
ax[0,1].imshow(Htrue)
ax[1,0].imshow(Wtrue)
ax[1,1].imshow(X)


## Algorithm 1: Alternating least squares (ALS)

# Initialize the factor matrices -- you can choose a better way if you have
Wals=np.dot(2,np.random.randn(I,K))
Hals=np.dot(2,np.random.randn(K,J))
MaxIterAls = 20;

# record the objective function values
obj_als = np.zeros((MaxIterAls,1))

for i in range(MaxIterAls):
    Wals = X.dot(Hals.transpose()).dot(np.linalg.inv(Hals.dot(Hals.transpose())))
    Hals = np.linalg.inv(Wals.transpose().dot(Wals)).dot(Wals.transpose()).dot(X)
    Xhat = Wals.dot(Hals)
    obj_als[i] = 0.5 * (np.linalg.norm(X - Xhat))**2
    # visualize the iterations
    # f, ax = plt.subplots(2,2)
    # ax[0,1].imshow(Hals)
    # ax[0,1].set_title('H' + str(i + 1))
    # ax[1,0].imshow(Wals)
    # ax[1,0].set_title('W' + str(i + 1))
    # ax[1,1].imshow(Xhat)
    # ax[1,1].set_title('X' + str(i + 1))

# draw the objective function
f, ax = plt.subplots(1,1)
plt.plot(obj_als,'r-o')
plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.title('ALS')




# Algorithm 2: Gradient Descent (GD)

# Initialize the factor matrices
I=10
J=20
K=3

Wgd = np.dot(2,np.random.randn(I,K))
Hgd = np.dot(2,np.random.randn(K,J))
MaxIterGd = 50

# record the objective function values
obj_gd = np.mat(np.zeros((MaxIterGd,1)))
# set the step size
eta = 0.01



for i in range(MaxIterGd):
    Wgd = Wgd + eta * (X - Wgd.dot(Hgd)).dot(Hgd.transpose())
    Hgd = Hgd + eta * Wgd.transpose().dot(X - Wgd.dot(Hgd))
    Xhat = Wgd.dot(Hgd)
    obj_gd[i] = 0.5 * (np.linalg.norm(X - Xhat))**2
    # visualize the iterations
    # f, ax = plt.subplots(2,2)
    # ax[0,1].imshow(Hgd)
    # ax[0,1].set_title('H' + str(i + 1))
    # ax[1,0].imshow(Wgd)
    # ax[1,0].set_title('W' + str(i + 1))
    # ax[1,1].imshow(Xhat)
    # ax[1,1].set_title('X' + str(i + 1))

# draw the objective function
f, ax = plt.subplots(1,1)
plt.plot(obj_gd,'r-o')
plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.title('GD')
plt.show()

