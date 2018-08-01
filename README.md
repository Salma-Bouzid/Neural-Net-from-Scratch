# Neural-Net-from-Scratch

Objective:  Train a neural network with 1 hidden layer to approximate the xor function
using quadratic loss.

## Model Architecture:
● A one hidden layer neural network .

● The hidden layer had 2 units.

● The chosen non linearity was sigmoid for the hidden layer.

● A bias term was added for both the hidden and output layer.

● The output layer was a pure affine transformation without any nonlinearity.

## Weight Initialization:
● The weight matrices were initialized with values from a normal distribution with mean of 0 and a standard deviation of 0.01.

● The bias terms were initialized with zeros.

## Training:
● Backpropagation was used to compute the gradients.

● Full batch gradient descent was used to update the weights and biases.

● The learning rate was set to 0.01.

● The training loss plateaued after the 30th epoch.
