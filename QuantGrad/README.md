
## File: `QuantGrad`

This file implements a basic autodifferentiation library to calculate gradients and visualize computation graphs. It provides:

- **Value Class**: Encapsulates data, gradients, and operations for scalar values.
- **Operations**: Supports arithmetic operations, activation functions (e.g., tanh), and backpropagation.
- **Visualization**: Generates computation graphs using Graphviz for understanding the flow of gradients.

---

## File: `NN`

This file builds a neural network using the `QuantGrad` library. It includes:

- **Neuron**: Implements a single neuron with weights, bias, and a tanh activation.
- **Layer**: A collection of neurons forming a single layer.
- **MLP (Multi-Layer Perceptron)**: A full feedforward neural network with customizable layers.
- **Training Loop**: Uses backpropagation and gradient descent to optimize the network.
- **Visualization**: Visualizes the loss computation graph using Graphviz.

---

Both files work together to create and train a simple neural network with a custom autodifferentiation engine.
