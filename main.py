import numpy as np
import matplotlib.pyplot as plt

from preprocessing import load_csv, preprocess_data
from mlp import MLP
from evaluation import Evaluator
from visualization import plot_mlp_decision_boundary

#Load and preprocess dataset
df = load_csv("penguins.csv")

data = preprocess_data(df)

#Prepare training and testing data
X_train = data["X_train_scaled"]
y_train = data["y_train_encoded"]

X_test = data["X_test_scaled"]
y_test_labels = data["y_test"]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

#Model training
np.random.seed(42)
model = MLP(
    hidden_layers=[10],
    learning_rate=0.03,
    epochs=1000,
    use_bias=False,
    activation_type="tanh"
)
loss_history = model.train(X_train, y_train)

#Testing
y_pred = model.predict(X_test)

#Evaluation
evaluator = Evaluator()
cm, accuracy = evaluator.conf_matrix(
    y_test_labels,
    y_pred,
    classes=(0, 1, 2)
)

print("\nConfusion Matrix:")
print(np.array(cm))

print("\nOverall Accuracy:", accuracy * 100, "%")

#Store training results
training_results = {
    "model": model,
    "X_train": data["X_train"],
    "X_test": data["X_test"],
    "X_train_scaled": data["X_train_scaled"],
    "X_test_scaled": data["X_test_scaled"],
    "y_train": data["y_train"],
    "y_test": data["y_test"],
    "mean": data["mean"],
    "std": data["std"]
}

#Visualization
fig = plot_mlp_decision_boundary(
    model=model,
    training_results=training_results,
    feature1="CulmenLength",
    feature2="FlipperLength",
    scale="Standardized"
)
plt.show()