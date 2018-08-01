
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

inputs = np.array([[0,0],
                   [0,1],
                   [1,0],
                   [1,1]])

y_true = np.array([0,
                   1,
                   1,
                   0]).reshape(-1,1)

def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return x*(1-x)


# ## Initialize random weight matrix and bias vector


w1 = np.random.normal(loc=0.0, scale=0.01, size=(2,2))  #initialize a random weight matrix 
b1 = np.zeros((1,2))  #initialize a random bias vector

w2 = np.random.normal(loc=0.0, scale=0.01, size=(2,1))
b2 = np.zeros((1,1))  # has to have the dimensions of the output


def forward_pass(w1,b1,w2,b2,inputs):
    z1 = np.dot(inputs, w1) +b1 
    a1 = sigmoid(z1)
    z2 = np.dot(a1,w2)+b2
    return z1, a1, z2


def compute_loss(z2, y_true):
    loss = ((y_true-z2)**2).mean(axis = 0, keepdims=True).flatten()
    return loss[0]


# X : inputs
# 
# m: X.shape[0]
# 
# w1: weight matrix of the first pass
# 
# b1: bias vector of the first pass
# 
# z1 = X*w1 +b1 : affine linearity of the first pass
# 
# a1: sigmoid(z1) : non linearity of the first pass
# 
# w2: weight matrix of the second pass
# b2 : bias vector of the second pass
# 
# z2 = w2* a1 +b2 :affine linearity of the second pass
# 
# dz2= dloss/dz2 = mean( 2 * (z2-y_true))
# dw2 = dloss/dw2 = dz2 * a1
# db2 = dloss/db2 = dz2
# 
# da1 = dloss/da1 = dloss/dz2 * dz2/ da1 = dz2 * w2  
# dz1 = dloss/dz1 = dloss/dz2 * dz2/da1 * da1/dz1 = da1 * z1 * (1-sigmoid(z1))
# 
# dw1 = dloss/dw1 = dloss/dz2 * dz2/da1 * da1/dz1 * dz1/dw1 = dz1 * X
# db1 = dloss/db1 = dloss/dz2 * dz2/da1 * da1/dz1 * dz1/db1 = dz1

# # backward pass : Compute gradients


def compute_gradients(inputs,w1,b1,z1,a1,w2,b2,z2):
        
    dz2 =  2*(z2-y_true)
    dw2 = np.dot(a1.T, dz2)
    db2 = dz2.sum(axis=0)
    
    da1 = dz2 * w2.T
    d_sigmoid = derivative_sigmoid(a1)
    
    dz1 = np.multiply(da1, d_sigmoid)
    dw1 = np.dot(dz1.T, inputs)
    db1 = dz1.sum(axis=0)

    return dw2, db2, dw1, db1


def update_parameters(inputs,w1,b1,z1,a1,w2,b2,z2, lr):
    
    dw2, db2, dw1, db1 = compute_gradients(inputs, w1, b1, z1,
                                           a1, w2, b2, z2)
    
    w1 = w1 - dw1 * lr
    b1 = b1 - db1 * lr
    
    w2 = w2 - dw2 * lr 
    b2 = b2 - db2 * lr
    
    return w1, b1, w2, b2 


def neural_network(inputs,w1,b1,w2,b2,epochs ):
    losses = []
    for epoch in range(epochs):
        # Forward pass
        z1,a1,z2 = forward_pass(w1,b1,w2,
                                b2,inputs)

        # Updating gradients
        w1, b1, w2, b2 = update_parameters(inputs,w1,b1,z1,a1,w2,b2,z2, lr)
        loss = compute_loss(z2, y_true)

        print("Loss at epoch {}: {}".format(epoch, loss))
        losses.append(loss)
        
    model = { "w1":w1,"b1": b1 , "w2":w2,"b2": b2 }
    

    return model, losses


lr = 0.01
epochs = 100
losses = neural_network(inputs,w1,b1,w2,b2,epochs )[1]



fig, ax = plt.subplots()
fig.set_size_inches(12, 7)
x_epoch = np.arange(epochs)
ax.plot(x_epoch,losses)
ax.set_xlabel("Number of Epochs")
ax.set_ylabel("Quadratic Loss ")
plt.show()
