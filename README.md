# CNN-FROM-SCRATCH

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Framework](https://img.shields.io/badge/Framework-NumPy-orange)

A fully from-scratch implementation of a Convolutional Neural Network (CNN) in Python using only NumPy.  
Includes convolutional layers, pooling, activation functions, optimizers, loss functions, dropout, batch normalization, and custom data generation.  
This project is designed for learning, experimenting, and extending convolutional network fundamentals without relying on high-level frameworks like TensorFlow or PyTorch.

---

## 1. Introduction

This project provides an educational and transparent implementation of a Convolutional Neural Network (CNN) built entirely from scratch using NumPy.

**Target audience:**
- Students learning how CNNs work internally
- Developers experimenting with custom low-level layers
- Researchers prototyping interpretable models with full control

---

## 2. Demo Screenshots

### 1. Feature Map Visualization  
_Example result of applying convolutional filters to synthetic data._  
_Image to be added: `notebook/images/conv_feature_map.png`_

### 2. Synthetic Dataset Classification  
_Custom 2D datasets (e.g., spiral, circle) processed through CNN layers._  
_Image to be added: `notebook/images/cnn_synthetic_result.png`_

---

## 3. Key Features

- **Layer Implementations:**
  - Convolutional layers (`Conv2D`) with custom stride, padding, and kernel
  - Pooling layers (`MaxPool2D`)
  - Fully-connected layers (`Linear`, `Flatten`)
  - Activation layers: ReLU, Sigmoid, Tanh, Leaky ReLU, Softmax
  - Dropout layer for regularization
  - Batch Normalization (`BatchNorm1D`, `BatchNorm2D`)

- **Loss Functions:**
  - Mean Squared Error (MSE)
  - Cross Entropy Loss
  - Softmax Cross Entropy (numerically stable)

- **Optimizers:**
  - SGD
  - Momentum
  - RMSProp
  - Adagrad
  - Adam

- **Training Utilities:**
  - Gradient clipping
  - Mini-batch iterator with shuffle
  - Model save/load using Pickle

- **Data Generation:**
  - Spiral, Circle, Line, Zone (2D/3D), and Polynomial regression
  - Support for CSV export and multi-class classification

---

## 4. System Requirements

- Python 3.8 or later  
- Required libraries:
  ```bash
  numpy>=1.21.0
  matplotlib>=3.4.0
  scikit-learn>=1.0.0
  tensorflow>=2.8.0  # (used only to load MNIST or external datasets)
````

---

## 5. Installation & Usage

Clone the repository:

```bash
git clone https://github.com/honggquan24/CNN-from-scratch.git
cd CNN-from-scratch
```

(Optional) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # On Linux/macOS
venv\Scripts\activate           # On Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 6. Folder Structure

```
CNN-from-scratch/
├── activations.py        # ReLU, Sigmoid, Tanh, LeakyReLU, Softmax
├── layers.py             # Conv2D, MaxPool2D, Linear, Flatten, Dropout, BatchNorm
├── loss.py               # MSE, CrossEntropy, SoftmaxCrossEntropy
├── network.py            # CNN container class with forward/backward/update
├── optimizers.py         # SGD, Momentum, RMSProp, Adagrad, Adam
├── utils.py              # Save/load, batch iterator, clip gradients
├── data_generator.py     # Spiral, Circle, Line, Zone, Zone_3D, Polynomial datasets
├── requirements.txt
└── notebook/
    └── images/           # Visualizations: conv_feature_map.png, cnn_synthetic_result.png, etc.
