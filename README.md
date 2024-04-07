# Neural Network for MNIST Classification

This Python project implements a neural network from scratch to classify handwritten digits using the MNIST dataset. It showcases the basics of neural network architecture, forward propagation, backpropagation, and the training process, making it an excellent resource for educational purposes.

The source code was taken from mnielsen/neural-networks-and-deep-learning and slightly modernized. Added comments in Russian to understand the logic of the neural network
## Features

- Efficient loading and preprocessing of the MNIST dataset.
- Implementation of a neural network with customizable architecture.
- Training the network with stochastic gradient descent.
- Evaluating the network's performance with accuracy metrics.

## Getting Started

### Prerequisites

Ensure Python 3.x is installed on your system. This project requires NumPy for numerical operations.

### Installation

1. Clone the repository to your local machine.

    ```bash
    git clone https://github.com/vemneyy/handwritten-digit-recognition
    ```

2. Navigate to the project directory and install the required packages using pip.

    ```bash
    pip install -r requirements.txt
    ```

### Usage

Run the main script to start the training process and evaluate the network on the MNIST dataset.

```bash
python main.py
```

## Files Description

- `main.py`: Initializes the network, loads the dataset, performs the training process, and evaluates the model's accuracy on test data.
- `mnist_loader.py`: Contains functions to download, load, and preprocess the MNIST dataset.
- `network.py`: Implements the neural network class, including methods for the forward pass, backpropagation, and updating model weights.