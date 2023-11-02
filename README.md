# Machine Learning Basics from Scratch

This repo will have all the code that I used to learn about neural networks and machine learning. Some implementations use no libraries and only basic mathematical operations and loops, while others use libraries to make calculations simpler.

## Table of Contents

- [Basic Gradient Descent](#gradient-descent)
- [Cost Function](#cost)
- [Shallow Neural Network](#ShallowNeuralNetwork)




## Gradient-Descent

This section is about how gradient descent is implemented/works. The code in UpdateWeightsandBiases file performs one step of the gradient descent algorithm. 
It calculates the gradients and the changes to be made to the weights and the bias.

The weight and bias update formula is:

```math
W_j = W_j - \alpha \frac{\partial J}{\partial W_j}
```
```math
b = b - \alpha \frac{\partial J}{\partial b}
```
Implemented as:
```math
W_j = W_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)}) x_j^{(i)}
```
```math
b = b - \alpha \frac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)})
```




## ShallowNeuralNetwork

I build this shallow neural network from scratch using Python. The implementation covers:

- Creating seperate Test and Training Samples
- Testing out activation functions with my own layer architecture
- Forward propagation
- Calculating loss and implementing backpropagation
- Updating weights and biases using gradient descent
- Finally evaluating this "from scratch" model and making predictions with the optimized weights and biases


