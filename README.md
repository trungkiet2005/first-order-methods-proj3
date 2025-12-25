# Optimization Techniques in Deep Learning

This project implements and compares various optimization algorithms for deep learning, including:
- Gradient Descent
- Momentum
- Nesterov Accelerated Gradient
- AdaGrad
- RMSProp
- Adam

## Project Structure

```
optimization-project/
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── optimizers_numpy.py
│   ├── optimizers_torch.py
│   ├── test_functions.py
│   └── models.py
├── utils/
│   ├── __init__.py
│   └── visualization.py
└── experiments/
    ├── __init__.py
    ├── run_math_functions.py
    └── run_mnist.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run Math Function Experiments
```bash
python experiments/run_math_functions.py
```

### Run MNIST Training
```bash
python experiments/run_mnist.py
```

## Features

- NumPy-based optimizers for mathematical test functions
- PyTorch-based optimizers for neural network training
- Test functions: Sphere, Rosenbrock, Beale
- Neural network models: MLP, SimpleCNN
- Visualization utilities for convergence analysis

