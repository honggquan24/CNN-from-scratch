# CNN-FROM-SCRATCH

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Framework](https://img.shields.io/badge/Framework-NumPy-orange)

A fully **from-scratch** implementation of a **Convolutional Neural Network (CNN)** in Python using only **NumPy**, complete with forward/backward propagation, convolutional layers, pooling layers, activation functions, optimizers, loss functions, and visualization support. This project is designed for learning, experimenting, and extending deep learning fundamentals without relying on high-level frameworks like TensorFlow or PyTorch.

---

## 1. Introduction

**Purpose:**  
The goal of this project is to provide a clear, educational implementation of a Convolutional Neural Network (CNN) from scratch.  
It is suitable for:
- Students learning computer vision and deep learning.
- Developers experimenting with custom CNN architectures.
- Researchers prototyping lightweight models without heavy dependencies.

### Demo Screenshots

#### 1. Image Classification
![CIFAR-10 Classification](notebook/images/cifar10_result.png)
> CIFAR-10 image classification results using the trained CNN.

#### 2. Feature Map Visualization
![Feature Maps](notebook/images/feature_maps.png)
> Visualization of learned convolutional filters and feature maps.

#### 3. Training Progress
![Training Loss](notebook/images/training_progress.png)
> Training and validation loss curves over epochs.

---

## 2. Key Features

- **Convolutional Layers**:
  - 2D Convolution (`Conv2D`) with configurable kernels, stride, and padding.
  - Optimized implementation using im2col for efficient matrix multiplication.
  - Support for multiple input/output channels.

- **Pooling Layers**:
  - Max Pooling (`MaxPool2D`) with configurable pool size and stride.
  - Optimized using NumPy stride tricks for better performance.

- **Fully Connected Layers**:
  - Dense (`Linear`) layers for classification heads.
  - Flatten layer to convert 4D tensors to 2D.

- **Activation Functions**:
  - ReLU, Sigmoid, Tanh, Leaky ReLU, Softmax.
  - Efficient implementations with proper gradient computation.

- **Regularization**:
  - Dropout for preventing overfitting.
  - Batch Normalization (1D and 2D) for stable training.

- **Loss Functions**:
  - Mean Squared Error (MSE) for regression.
  - Cross Entropy Loss with integrated Softmax for classification.
  - Softmax Cross Entropy Loss for efficient training.

- **Optimizers**:
  - SGD, Momentum, RMSProp, Adagrad, Adam.
  - Proper parameter updates for convolutional layers.

- **Data Generation**:
  - Spiral, Circle, Line, and Zone datasets for 2D visualization.
  - 3D data generators for complex classification tasks.
  - Polynomial data generation for regression tasks.

- **Training Utilities**:
  - Gradient clipping for stable training.
  - Mini-batch iterator with shuffling support.
  - Model save/load functionality with complete state preservation.

---

## 3. System Requirements

- **Python**: 3.8+
- **Required Libraries**:
  ```bash
  numpy>=1.21.0
  matplotlib>=3.4.0
  scikit-learn>=1.0.0
  ```

---

## 4. Installation & Usage

```bash
git clone https://github.com/honggquan24/CNN-from-scratch.git
cd CNN-FROM-SCRATCH

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install requirements
pip install -r requirements.txt
```

### Quick Start Example

```python
from core import *
import numpy as np

# Create a simple CNN for MNIST-like data
model = CNN()

# Add convolutional layers
model.add_layer(Conv2D(in_channels=1, out_channels=32, kernel_size=3, padding=1))
model.add_layer(ReLULayer())
model.add_layer(MaxPool2D(pool_size=2))

model.add_layer(Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1))
model.add_layer(ReLULayer())
model.add_layer(MaxPool2D(pool_size=2))

# Add fully connected layers
model.add_layer(Flatten())
model.add_layer(Linear(input_size=64*7*7, output_size=128))
model.add_layer(ReLULayer())
model.add_layer(Dropout(dropout_rate=0.5))
model.add_layer(Linear(input_size=128, output_size=10))

# Create optimizer and loss function
optimizer = Adam(learning_rate=0.001)
loss_fn = SoftmaxCrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in batch_iterator(train_X, train_y, batch_size=32):
        # Forward pass
        predictions = model.forward(batch_x)
        loss = loss_fn(predictions, batch_y)
        
        # Backward pass
        grad = loss_fn.backward()
        model.backward(grad)
        
        # Update parameters
        model.update_params(optimizer)
        model.zero_grad()
```

---

## 5. Project Structure

```
CNN-FROM-SCRATCH/
│
├── core/                     # Core implementation
│   ├── __init__.py          # Package initialization
│   ├── activations.py       # Activation functions (ReLU, Sigmoid, etc.)
│   ├── layers.py            # Layer implementations
│   │   ├── Linear           # Fully connected layer
│   │   ├── Conv2D           # 2D Convolutional layer
│   │   ├── MaxPool2D        # Max pooling layer
│   │   ├── Flatten          # Tensor flattening
│   │   ├── Dropout          # Dropout regularization
│   │   ├── BatchNorm1D/2D   # Batch normalization
│   │   └── Activation layers # ReLU, Sigmoid, Tanh, etc.
│   ├── loss.py              # Loss functions
│   │   ├── MSE              # Mean Squared Error
│   │   ├── CrossEntropy     # Cross Entropy Loss
│   │   └── SoftmaxCrossEntropy # Combined Softmax + CrossEntropy
│   ├── network.py           # CNN and NeuralNetwork classes
│   ├── optimizers.py        # Optimization algorithms
│   │   ├── SGD              # Stochastic Gradient Descent
│   │   ├── Momentum         # SGD with Momentum
│   │   ├── RMSProp          # RMSProp optimizer
│   │   ├── Adagrad          # Adaptive Gradient
│   │   └── Adam             # Adam optimizer
│   ├── utils.py             # Utility functions
│   │   ├── save/load        # Model serialization
│   │   ├── batch_iterator   # Data batching
│   │   └── gradient_clipping # Gradient utilities
│   └── data_generator.py    # Dataset generators
│       ├── Spiral           # 2D spiral classification
│       ├── Circle/Line      # Geometric datasets
│       ├── Zone/Zone_3D     # Multi-class datasets
│       └── Polynomial       # Regression datasets
│
├── examples/                 # Example implementations
│   ├── image_classification.py # CIFAR-10/MNIST examples
│   ├── feature_visualization.py # Feature map visualization
│   └── custom_architectures.py # Different CNN architectures
│
├── notebook/                 # Jupyter notebooks
│   ├── CNN_Tutorial.ipynb   # Step-by-step CNN tutorial
│   ├── Architecture_Comparison.ipynb # Different architectures
│   └── Feature_Visualization.ipynb # Visualization techniques
│
├── tests/                    # Unit tests
│   ├── test_layers.py       # Test convolutional layers
│   ├── test_activations.py  # Test activation functions
│   ├── test_optimizers.py   # Test optimization algorithms
│   └── test_utils.py        # Test utility functions
│
├── README.md
├── requirements.txt
└── setup.py
```

---

## 6. Architecture Components

### Convolutional Layers
- **Conv2D**: Efficient 2D convolution using im2col transformation
- **Optimized Implementation**: Matrix multiplication for fast computation
- **Flexible Configuration**: Customizable kernel size, stride, padding, and channels

### Pooling Layers  
- **MaxPool2D**: Downsampling with max pooling operation
- **Stride Tricks**: NumPy optimization for efficient pooling
- **Gradient Preservation**: Proper backpropagation through pooling

### Normalization
- **BatchNorm2D**: Batch normalization for convolutional features
- **BatchNorm1D**: Batch normalization for fully connected layers
- **Running Statistics**: Proper handling of training/inference modes

### Advanced Features
- **Dropout**: Regularization to prevent overfitting
- **Multiple Optimizers**: Adam, RMSProp, Momentum, etc.
- **Flexible Architecture**: Easy to add new layer types

---

## 7. Performance Optimizations

- **im2col Transformation**: Converts convolution to efficient matrix multiplication
- **Vectorized Operations**: Leverages NumPy's optimized routines
- **Memory Efficient**: Proper gradient computation and caching
- **Batch Processing**: Efficient mini-batch training support

---

## 8. Educational Value

This implementation provides:
- **Clear Code Structure**: Easy to understand and modify
- **Mathematical Foundations**: Proper gradient derivations
- **Debugging Support**: Comprehensive error handling
- **Extensibility**: Simple to add new features

---

## 9. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 10. Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 11. Acknowledgments

- Inspired by CS231n: Convolutional Neural Networks for Visual Recognition
- NumPy documentation and optimization techniques
- Deep learning community for best practices and implementations