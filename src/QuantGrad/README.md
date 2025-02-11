
## File: `QuantGrad`

This file implements a basic autodifferentiation library to calculate gradients and visualize computation graphs. It provides:

- **Value Class**: Encapsulates data, gradients, and operations for scalar values.
- **Operations**: Supports arithmetic operations, activation functions (e.g., tanh), and backpropagation.
- **Visualization**: Generates computation graphs using Graphviz for understanding the flow of gradients.


### Example Usage

Here's an example of how to use the `QuantGrad` library to perform basic operations and visualize the computation graph:

```python
from QuantGrad import Value, draw_dot

# Create scalar values
a = Value(2.0, _label='a')
b = Value(-3.0, _label='b')
c = Value(10.0, _label='c')

# Perform operations
d = a * b + c
e = a.tanh()
f = d * e

# Perform backpropagation
f.backwards()

# Visualize the computation graph
dot = draw_dot(f)
dot.render('computation_graph', format='png')
```

This example demonstrates creating scalar values, performing arithmetic and activation operations, and visualizing the resulting computation graph using Graphviz.

---

## File: `NN`

This file builds a neural network using the `QuantGrad` library. It includes:

- **Neuron**: Implements a single neuron with weights, bias, and a tanh activation.
- **Layer**: A collection of neurons forming a single layer.
- **MLP (Multi-Layer Perceptron)**: A full feedforward neural network with customizable layers.
- **Training Loop**: Uses backpropagation and gradient descent to optimize the network.
- **Visualization**: Visualizes the loss computation graph using Graphviz.

---

## Functionality Overview

### Value Class

The `Value` class is the core of the `QuantGrad` library. It encapsulates a scalar value and its gradient, and supports various operations. Key functionalities include:

- **Initialization**: Create a `Value` object with an initial scalar value and an optional label.
- **Arithmetic Operations**: Supports addition, subtraction, multiplication, and division.
- **Activation Functions**: Includes the hyperbolic tangent (`tanh`) function.
- **Backpropagation**: Computes gradients for all operations leading to the current value.

### Operations

The library supports a variety of operations on `Value` objects:

- **Addition**: `a + b`
- **Subtraction**: `a - b`
- **Multiplication**: `a * b`
- **Division**: `a / b`
- **Tanh Activation**: `a.tanh()`

These operations are overloaded to ensure that gradients are tracked and computed correctly during backpropagation.

### Visualization

The `QuantGrad` library provides tools to visualize computation graphs using Graphviz. This helps in understanding the flow of gradients through the network.

- **draw_dot**: Generates a Graphviz dot representation of the computation graph for a given `Value`.

### Neuron

The `Neuron` class represents a single neuron in a neural network. It includes:

- **Weights and Bias**: Each neuron has weights and a bias term.
- **Activation**: Uses the `tanh` function as the activation function.
- **Forward Pass**: Computes the output of the neuron given an input.

### Layer

The `Layer` class is a collection of neurons. It supports:

- **Initialization**: Create a layer with a specified number of input and output neurons.
- **Forward Pass**: Computes the outputs of all neurons in the layer given an input.

### MLP (Multi-Layer Perceptron)

The `MLP` class represents a full feedforward neural network. It includes:

- **Initialization**: Create an MLP with customizable layers.
- **Forward Pass**: Computes the output of the network given an input.
- **Training**: Uses backpropagation and gradient descent to optimize the network.

### Training Loop

The training loop is responsible for optimizing the neural network. It includes:

- **Loss Calculation**: Computes the loss for a given set of predictions and targets.
- **Backpropagation**: Computes gradients for all parameters in the network.
- **Gradient Descent**: Updates the parameters using the computed gradients.

### Visualization

The library also provides tools to visualize the loss computation graph using Graphviz, aiding in understanding the optimization process.

- **draw_dot**: Generates a Graphviz dot representation of the loss computation graph.

---

Both files work together to create and train a simple neural network with a custom autodifferentiation engine.
