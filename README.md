# Penguins Classification Using Multi-Layer Perceptron (MLP)

## 1. Project Overview

This project implements a **Multi-Layer Perceptron (MLP)** neural network from scratch using Python and NumPy to classify penguin species based on their physical characteristics.

The project includes data preprocessing, neural network training, model evaluation, and visualization through an interactive Streamlit web interface.

The goal is to classify each penguin into one of three species:

- Adelie
- Chinstrap
- Gentoo

## 2. Dataset

The project uses the `penguins_pr.csv` dataset, which contains 150 samples.

Each species has 50 samples.

### Input Features

The model uses five input features:

1. Culmen Length
2. Culmen Depth
3. Flipper Length
4. Origin Location
5. Body Mass

### Output Classes

| Species | Class Label |
|---|---:|
| Adelie | 0 |
| Chinstrap | 1 |
| Gentoo | 2 |

### Dataset Splitting

The dataset is divided into training and testing sets:

- **Training:** 30 samples per class (90 samples total).
- **Testing:** 20 samples per class (60 samples total).

A fixed random seed is used to ensure reproducibility.

## 3. Project Features

### Data Preprocessing
- Load the dataset from a CSV file.
- Handle missing values.
- Encode categorical features and target labels.
- Split the dataset into training and testing sets.
- Apply feature standardization using training-set statistics.

### MLP Neural Network
- Implemented from scratch using NumPy.
- Supports dynamically configurable hidden layers.
- Supports configurable neurons in each hidden layer.
- Supports Sigmoid and Tanh activation functions.
- Supports enabling or disabling bias.
- Configurable learning rate and number of training epochs.
- Implements feedforward propagation and backpropagation.

### Model Evaluation
- Predict penguin species using the trained model.
- Generate a confusion matrix.
- Calculate overall classification accuracy.

### Visualization
- Display dataset information and class distribution.
- Visualize the training loss curve.
- Display the confusion matrix.
- Visualize the decision boundary using two selected features.
- Display final trained weights and bias values.

## 4. Project Structure

```text
Task2/
│
├── app.py                  # Streamlit application
├── preprocessing.py        # Data loading and preprocessing
├── mlp.py                  # MLP neural network implementation
├── evaluation.py           # Model evaluation and confusion matrix
├── visualization.py        # Decision boundary visualization
├── penguins_pr.csv         # Dataset
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

*Note: Adjust the filenames above if your actual project uses a different module structure.*

## 5. Requirements

The project requires:

- Python 3.10 or later
- NumPy
- Pandas
- Matplotlib
- Streamlit

## 6. Installation

### Step 1: Clone or Download the Project

Download the project and navigate to its directory:

```bash
cd Task2
```

### Step 2: Create a Virtual Environment

Create an isolated Python environment:

```bash
python3 -m venv .venv
```

### Step 3: Activate the Virtual Environment

On Ubuntu/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### Step 4: Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install all project requirements:

```bash
pip install -r requirements.txt
```

## 7. Running the Application

After installing the dependencies and activating the virtual environment, run:

```bash
streamlit run app.py
```

Streamlit will start a local development server.

Open the following address in your browser:

```text
http://localhost:8501
```

## 8. Using the Application

1. Open the Streamlit application.
2. Review the dataset overview and class distribution.
3. Configure the neural network architecture:
   - Select the number of hidden layers.
   - Specify the number of neurons in each hidden layer.
4. Configure the training parameters:
   - Learning rate.
   - Number of epochs.
   - Activation function.
   - Bias option.
5. Click the **Train Model** button.
6. Review the training results:
   - Overall accuracy.
   - Training loss curve.
   - Confusion matrix.
   - Final model weights.
   - Final bias values.
7. Explore the decision boundary by selecting two input features and the desired data scale.

## 9. Requirements File

The `requirements.txt` file should contain:

```text
numpy
pandas
matplotlib
streamlit
```

## 10. Notes

- The MLP algorithm is implemented from scratch without using high-level deep learning frameworks such as TensorFlow or PyTorch.
- NumPy is used for numerical calculations.
- The training and testing split maintains 30 training samples and 20 testing samples for each species.
- Standardization parameters are calculated from the training data to avoid data leakage.
- A fixed random seed is used where applicable to improve reproducibility.

## 11. Deactivating the Virtual Environment

When you finish working, you can deactivate the virtual environment:

```bash
deactivate
```

---

**Developed as part of the Machine Learning Task – Penguin Species Classification using MLP.**
