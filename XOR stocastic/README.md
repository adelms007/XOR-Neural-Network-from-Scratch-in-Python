# XOR Neural Network from Scratch in Python

A minimalist implementation of a 3-neuron feedforward neural network built entirely **from scratch** using only standard Python and NumPy. This project demonstrates the fundamental mechanics of deep learning by implementing forward propagation, analytical backpropagation (using the Chain Rule), and Stochastic Gradient Descent (SGD) manually without any deep learning frameworks (like PyTorch or TensorFlow).

## 🚀 Overview

The goal of this network is to solve the classic **XOR (Exclusive OR)** problem. Because the XOR function is not linearly separable, a single-layer perceptron cannot solve it. This implementation uses a Multi-Layer Perceptron (MLP) architecture with non-linear activation functions to successfully learn the XOR logic gates.

### Architecture
- **Input Layer**: 2 inputs + 1 Bias input (permanently set to `1`).
- **Hidden Layer**: 2 hidden neurons (`neuron_1`, `neuron_2`) using the **Sigmoid** activation function.
- **Output Layer**: 1 output neuron (`neuron_3`) using the **Sigmoid** activation function to output a probability between 0 and 1.

---

## 🧠 Mathematical Mechanics Implemented

### 1. Forward Propagation
For each neuron, the input vector is combined with the weight vector using a dot product, plus the bias, and then passed through the Sigmoid activation function:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### 2. Analytical Backpropagation
The gradients are calculated explicitly for each individual weight by applying the **Chain Rule** backward from the output layer to the hidden layer. 

For the Sigmoid function, its derivative is computed using its output:
$$\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$$

- **Output Error ($\delta_{out}$)**: Derived from the Mean Squared Error (MSE) cost function.
- **Hidden Errors ($\delta_{h_1}, \delta_{h_2}$)**: Backpropagated from the output layer through the connecting weights.

### 3. Weight Updates (SGD)
Weights are updated immediately after processing each training sample (Stochastic Gradient Descent) according to the learning rate ($\alpha$):
$$W_{new} = W_{old} - \alpha \cdot \frac{\partial J}{\partial W}$$

---

## 💻 Getting Started

### Prerequisites
Make sure you have `numpy` installed in your environment
