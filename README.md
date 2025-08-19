# The effect of activation functions on convergence in neural networks with low sample datasets

This project is an experimental deep learning setup designed to explore the training dynamics of neural networks on the MNIST handwritten digit classification dataset. It investigates the effects of different activation functions, reproducibility settings, and evaluation techniques on model performance.

## Overview

The notebook loads the MNIST dataset, trains models under varying conditions, and visualizes results to understand how architectural choices affect learning dynamics. Key aspects include:

- **Dataset Handling**: Uses `torchvision.datasets.MNIST` for data loading and `torch.utils.data.random_split` for creating subsets when needed.
- **Reproducibility**: Implements a custom `set_seed(seed)` function to ensure deterministic behavior in PyTorch and NumPy for fair comparisons across runs.
- **Model Training**: Tests multiple activation functions and tracks loss during training.
- **Evaluation**: Tracks loss curves, regression fits, and other performance metrics.

## Features

### Dataset Handling
- Loads MNIST training and test sets via `torchvision.datasets.MNIST`.
- Splits data into subsets for experimentation using `torch.utils.data.random_split`.

### Reproducibility
- Ensures determinism with PyTorch and NumPy for consistent results across runs and ensures robustness by manually setting the seed to either 42 or 451.

### Model Training
- Explores various activation functions
- Tracks per-epoch and per-batch loss to compare learning dynamics.

### Evaluation
- Plots loss curves, regression fits, and performance metrics for side-by-side comparisons.

### Experimentation
- Varies seeds, subsets of training data, and activation families.
- Collects results for comparative analysis.

## Results

### Loss Curves
- Plotted for each activation function to compare convergence behavior.

### Batch Dynamics
- Tracks per-batch loss to highlight differences in activation functions early in training.

## Findings & Reflections on exceptional cases

Within both models, the SeLU function consistly started at a higher training loss however normalised and consistly let to one of the lowest train and test losses in the end. This is due to SeLU's self normalisation as it exponentially punishes worse inputs, leading to better convergence and a higher loss with random weights due to the initialisation. 
Due to the low amount of epoches, overfitting is not prevalent and overall is not seen across activation functions.

### Multi-layer perception model

Within the MLP model, the models using the sigmoid functions had a mostly flat train and test loss regression line. This is likely due to oversaturation as the inputs might have been made into values close to (0,1) and in those regions, the derivative of sigmoid is almost 0, so backpropagation updates vanish. With little data, your weight updates don’t get enough variation to escape that saturation.

### TinyVGG

Sigmoid is still stagnant. Furthermore as seen in the batch update loss graphs, multiple functions however mainly SeLU explode for one batch update, not affecting the overall training loss and test loss.

## How to Run

1. Clone this repository or copy the notebook.
2. Install dependencies:
   ```bash
   pip install torch torchvision torchmetrics matplotlib tqdm