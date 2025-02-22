# Mini-Library of functions useful for Machine Learning Applications

## Implementation of the micrograd engine

The underlying processor or the computer itself doesn't have an explicit understanding of what differentiation is. 
The Backpropagation Algorithm being of extreme importance to training neural networks, relies on differentiation to perform gradient descent.
Understanding it mathematically allows one to cleverly define implicit methods to allow the machine to perform differentiation.

The Auto-grad engine creates a computational graph at run-time. This graph is a directed acyclic graph and can be sorted topologically. 
After generating this graph, the engine explores it backwards from the rightmost node with each node having the gradient set to the derivative with respect to the rightmost node.

Using pytorch, the same can be done by the following code:

```python
optimizer.zero_grad()
loss = ( y_actual - y_preds ) ** 2 / n # Assuming the loss function is the mean squared loss function.
loss.backward()
optimizer.step()
``` 

- 'optimizer.zero_grad()' sets the value of all the gradients to 0.0
- The loss is calculated using any of the loss functions defined by the programmer.
- 'loss.backward()' computes the gradients.
- 'optimizer.step()' updates the model parameters.

---

## Class for buiding neural networks

I've always built neural networks using the functionality provided by Pytorch, and this abstracts away the underlying complexities.
Thereby making it easier for the programmer to build their applications.

Building MLPs for any task is possible using the Linear class defined in 'arch.py'. The same goes for image processing using CNNs, this can be done using the Conv2D class defined in the same file. 
Although these are very basic and un-optimized implementations of the actual ones present in the Pytorch codebase, they aim to capture the general idea of the working of the commonly used architectures when building machine learning applications.

## Installation

Clone the repository:

   ```bash
   git clone https://github.com/R2D2-08/minilib.git
   cd minilib

## References 

1. A video by Andrej Karpathy on building micrograd and introduction to neural networks[The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)
