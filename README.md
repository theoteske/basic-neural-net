# Basic Neural Net: A Neural Network Implementation from Scratch

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-brightgreen.svg)](https://numpy.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/neural-net)](https://github.com/yourusername/neural-net/commits/main)

A NumPy-based neural network implementation built from scratch for educational purposes. This project implements core deep learning components including various layer types, activation functions, loss functions, and optimizers.

## Features

- **Layers**: Dense (fully connected) layers
- **Activation Functions**: ReLU, Sigmoid, Softmax, Identity
- **Loss Functions**: MSE, Cross-Entropy
- **Optimizers**: SGD with Momentum, Nesterov AGD, RMSprop, Adam
- **Sequential Model API**: Easy model construction and training

## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/yourusername/neural-net.git
cd neural-net
pip install -e .
```

To run the examples, install additional dependencies:

```bash
pip install -e ".[examples]"
```

## Usage

Basic example of creating and training a model:

```python
from neural_net import Sequential
from neural_net import layers, activations, losses, optimizers

# Create model
model = Sequential()
model.add(layers.InputLayer(input_shape=784))
model.add(layers.Dense(784, 128, activation=activations.Sigmoid))
model.add(layers.Dense(128, 10, activation=activations.Softmax))

# Train model
model.train(
    X_train,
    y_train,
    epochs=10,
    learning_rate=0.001,
    batch_size=32,
    loss=losses.CrossEntropy
)

# Make predictions
predictions = model.predict(X_test)
```

See the `examples` directory for more detailed usage examples, including MNIST digit classification.

## Project Structure

```
basic-neural-net/
├── basic_neural_net/     # Source code
│   ├── activations.py    # Activation functions
│   ├── layers.py         # Layer implementations
│   ├── losses.py         # Loss functions
│   ├── optimizers.py     # Optimization algorithms
│   └── nn.py             # Sequential model API
├── examples/             # Example notebooks
│   └── mnist_example.ipynb
└── requirements.txt      # Project dependencies
```

## Examples

The repository includes a Jupyter notebook demonstrating MNIST digit classification:

```bash
cd examples
jupyter notebook mnist_example.ipynb
```

## Requirements

- Python 3.7+
- NumPy
- SciPy

Optional dependencies for examples:
- TensorFlow (for MNIST dataset)
- Matplotlib (for visualizations)
- Jupyter (for notebooks)

## Contributing

Feel free to open issues or submit pull requests for improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{neural_net2025,
  author = {[Theo Teske]},
  title = {Basic Neural Net},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/theoteske/basic-neural-net}
}
```