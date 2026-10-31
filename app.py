
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import (
    load_csv,
    preprocess_data,
    FEATURES,
    SPECIES_MAPPING
)

from mlp import MLP
from evaluation import Evaluator
from visualization import plot_mlp_decision_boundary


# ==========================================
# 1. Page Configuration
# ==========================================

st.set_page_config(
    page_title="Penguins MLP Classifier",
    page_icon="🐧",
    layout="wide"
)

st.title("🐧 Penguins Species Classification")

st.markdown(
    "### Multi-Layer Perceptron (MLP) Neural Network"
)

st.write(
    "Train a neural network to classify penguin species "
    "using physical measurements and origin location."
)

st.divider()


# ==========================================
# 2. Load Dataset
# ==========================================

@st.cache_data
def get_dataset():
    return load_csv("penguins.csv")


try:
    df = get_dataset()

except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()


# ==========================================
# 3. Dataset Overview
# ==========================================

st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Samples", len(df))

with col2:
    st.metric("Input Features", len(FEATURES))

with col3:
    st.metric("Output Classes", len(SPECIES_MAPPING))


with st.expander("View Dataset"):
    st.dataframe(df, use_container_width=True)


with st.expander("View Class Distribution"):
    st.bar_chart(df["Species"].value_counts())


st.divider()


# ==========================================
# 4. MLP Configuration
# ==========================================

st.header("2. Configure MLP Model")

# ------------------------------------------
# Dynamic Hidden Layers Configuration
# ------------------------------------------

st.subheader("Network Architecture")

num_hidden_layers = st.slider(
    "Number of Hidden Layers",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

st.write("Configure the number of neurons in each hidden layer.")

hidden_layers = []

cols = st.columns(2)

for i in range(num_hidden_layers):

    with cols[i % 2]:

        default_neurons = 8 if i == 0 else 6

        neurons = st.number_input(
            f"Hidden Layer {i + 1} Neurons",
            min_value=1,
            max_value=128,
            value=default_neurons,
            step=1,
            key=f"hidden_layer_{i}"
        )

        hidden_layers.append(int(neurons))


# ------------------------------------------
# Display Architecture
# ------------------------------------------

architecture = [5] + hidden_layers + [3]

st.info(
    "Network Architecture: "
    + " → ".join(map(str, architecture))
)

st.divider()


# ------------------------------------------
# Training Parameters
# ------------------------------------------

st.subheader("Training Parameters")

col1, col2 = st.columns(2)

with col1:

    learning_rate = st.number_input(
        "Learning Rate",
        min_value=0.0001,
        max_value=1.0,
        value=0.01,
        step=0.001,
        format="%.4f"
    )

    epochs = st.number_input(
        "Number of Epochs",
        min_value=1,
        max_value=10000,
        value=1000,
        step=100
    )


with col2:

    activation = st.selectbox(
        "Activation Function",
        ["sigmoid", "tanh"]
    )

    use_bias = st.checkbox(
        "Enable Bias",
        value=False
    )


st.divider()


# ==========================================
# 5. Train Model
# ==========================================

train_button = st.button(
    "🚀 Train Model",
    type="primary",
    use_container_width=True
)


if train_button:

    with st.spinner("Training MLP model..."):

        try:

            # ----------------------------------
            # Preprocessing
            # ----------------------------------

            data = preprocess_data(df)

            X_train = data["X_train_scaled"]
            y_train = data["y_train_encoded"]

            X_test = data["X_test_scaled"]
            y_test = data["y_test"]


            # ----------------------------------
            # Model Initialization
            # ----------------------------------

            np.random.seed(42)

            model = MLP(
                hidden_layers=hidden_layers,
                learning_rate=learning_rate,
                epochs=int(epochs),
                use_bias=use_bias,
                activation_type=activation
            )


            # ----------------------------------
            # Training
            # ----------------------------------

            loss_history = model.train(
                X_train,
                y_train
            )


            # ----------------------------------
            # Prediction
            # ----------------------------------

            y_pred = model.predict(X_test)


            # ----------------------------------
            # Evaluation
            # ----------------------------------

            evaluator = Evaluator()

            cm, accuracy = evaluator.conf_matrix(
                y_test,
                y_pred,
                classes=(0, 1, 2)
            )


            # ----------------------------------
            # Store Results
            # ----------------------------------

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


            st.session_state["model"] = model

            st.session_state["training_results"] = training_results

            st.session_state["loss_history"] = loss_history

            st.session_state["confusion_matrix"] = cm

            st.session_state["accuracy"] = accuracy

            st.session_state["architecture"] = architecture.copy()


            st.success("Model trained successfully!")


        except Exception as e:

            st.error(f"Training failed: {e}")


# ==========================================
# 6. Display Training Results
# ==========================================

if "model" in st.session_state:

    st.divider()

    st.header("3. Training Results")

    model = st.session_state["model"]

    accuracy = st.session_state["accuracy"]

    loss_history = st.session_state["loss_history"]


    # --------------------------------------
    # Metrics
    # --------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Test Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Training Epochs",
            len(loss_history)
        )

    with col3:

        st.metric(
            "Final Training Loss",
            f"{loss_history[-1]:.6f}"
        )


    # --------------------------------------
    # Architecture Information
    # --------------------------------------

    st.subheader("Trained Architecture")

    st.code(
        " → ".join(
            map(str, st.session_state["architecture"])
        )
    )


    # --------------------------------------
    # Loss Curve
    # --------------------------------------

    st.subheader("Training Loss Curve")

    loss_df = pd.DataFrame({

        "Epoch": range(1, len(loss_history) + 1),

        "Loss": loss_history

    })

    st.line_chart(
        loss_df.set_index("Epoch"),
        use_container_width=True
    )


    # ======================================
    # 7. Model Evaluation
    # ======================================

    st.divider()

    st.header("4. Model Evaluation")

    cm = st.session_state["confusion_matrix"]

    class_names = [
        name for name, label
        in sorted(
            SPECIES_MAPPING.items(),
            key=lambda item: item[1]
        )
    ]

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

    st.subheader("Confusion Matrix")

    st.write(
        "Rows represent actual classes, "
        "and columns represent predicted classes."
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )


    st.subheader("Overall Accuracy")

    st.progress(float(accuracy))

    st.write(
        f"Accuracy: {accuracy * 100:.2f}%"
    )


    # ======================================
    # 8. Decision Boundary
    # ======================================

    st.divider()

    st.header("5. Decision Boundary Visualization")

    training_results = st.session_state["training_results"]


    col1, col2, col3 = st.columns(3)

    with col1:

        feature1 = st.selectbox(
            "X-axis Feature",
            FEATURES,
            index=0
        )

    with col2:

        feature2 = st.selectbox(
            "Y-axis Feature",
            FEATURES,
            index=2
        )

    with col3:

        scale = st.selectbox(
            "Feature Scale",
            ["Standardized", "Original"]
        )


    if feature1 == feature2:

        st.warning(
            "Please select two different features."
        )

    else:

        fig = plot_mlp_decision_boundary(
            model=model,
            training_results=training_results,
            feature1=feature1,
            feature2=feature2,
            scale=scale
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    # ======================================
    # 9. Model Details
    # ======================================

    st.divider()

    with st.expander("View Model Details"):

        st.write("Input Neurons: 5")

        st.write(
            f"Hidden Layers: {len(model.hidden_layers)}"
        )

        for i, neurons in enumerate(model.hidden_layers):

            st.write(
                f"Hidden Layer {i + 1}: {neurons} neurons"
            )

        st.write("Output Neurons: 3")

        st.write(
            f"Activation Function: {model.activation_type}"
        )

        st.write(
            f"Learning Rate: {model.learning_rate}"
        )

        st.write(
            f"Bias Enabled: {model.use_bias}"
        )

    # =======================================================
    # 10. FINAL MODEL WEIGHTS 
    # =======================================================

    st.subheader("⚖️ Final Model Weights")

    with st.expander("View Final Weights", expanded=False):

        for i, layer_weights in enumerate(model.weight_list):

            st.markdown(
                f"**Layer {i + 1}: "
                f"{layer_weights.shape[0]} → {layer_weights.shape[1]}**"
            )

            weights_df = pd.DataFrame(
                layer_weights,
                index=[
                    f"Neuron {j + 1}"
                    for j in range(layer_weights.shape[0])
                ],
                columns=[
                    f"Neuron {j + 1}"
                    for j in range(layer_weights.shape[1])
                ]
            )

            st.dataframe(
                weights_df.style.format("{:.6f}"),
                use_container_width=True
            )

    # =======================================================
    # 11. FINAL Bias WEIGHTS 
    # =======================================================
    st.subheader("Final Bias Values")

    with st.expander("View Biases", expanded=False):

        for i, layer_bias in enumerate(model.bias_list):

            st.markdown(f"**Layer {i + 1} Bias**")

            bias_values = np.asarray(layer_bias).flatten()

            bias_df = pd.DataFrame({
                "Neuron": [
                    f"Neuron {j + 1}"
                    for j in range(len(bias_values))
                ],
                "Bias": bias_values
            })

            st.dataframe(
                bias_df.style.format({"Bias": "{:.6f}"}),
                use_container_width=True
            )

    # =======================================================
    # 12. SINGLE SAMPLE PREDICTION 
    # =======================================================
    st.divider()
    st.header("🐧 Classify a New Penguin")

    st.write(
        "Enter the penguin's measurements below, "
        "and the trained MLP model will predict its species."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            culmen_length = st.number_input(
                "Culmen Length (mm)",
                min_value=0.0,
                value=45.0,
                step=0.1
            )

            culmen_depth = st.number_input(
                "Culmen Depth (mm)",
                min_value=0.0,
                value=17.0,
                step=0.1
            )

            flipper_length = st.number_input(
                "Flipper Length (mm)",
                min_value=0.0,
                value=200.0,
                step=1.0
            )

        with col2:
            origin_location = st.selectbox(
                "Origin Location",
                options=["Torgersen", "Biscoe", "Dream"]
            )

            body_mass = st.number_input(
                "Body Mass (g)",
                min_value=0.0,
                value=4000.0,
                step=10.0
            )

        predict_button = st.form_submit_button(
            "Classify Penguin",
            use_container_width=True
        )


    if predict_button:

        # Encode origin location
        origin_mapping = {
            "Torgersen": 0,
            "Biscoe": 1,
            "Dream": 2
        }

        # Create input sample in the same feature order used during training
        sample = np.array([[
            culmen_length,
            culmen_depth,
            flipper_length,
            origin_mapping[origin_location],
            body_mass
        ]], dtype=float)

        # Retrieve training preprocessing parameters
        training_results = st.session_state["training_results"]

        mean = np.asarray(training_results["mean"]).flatten()
        std = np.asarray(training_results["std"]).flatten()

        # Standardize using training mean and standard deviation
        sample_scaled = (sample - mean) / std

        # Predict using the trained model
        model = st.session_state["model"]
        prediction = model.predict(sample_scaled)

        # Decode predicted class
        species_mapping = {
            0: "Adelie",
            1: "Chinstrap",
            2: "Gentoo"
        }

        predicted_class = int(prediction[0])
        predicted_species = species_mapping[predicted_class]

        # Display result
        st.success("Classification completed!")

        st.subheader("Prediction Result")

        st.markdown(
            f"### 🐧 Predicted Species: **{predicted_species}**"
        )

        st.write("Input measurements:")

        input_df = pd.DataFrame(
            sample,
            columns=[
                "Culmen Length",
                "Culmen Depth",
                "Flipper Length",
                "Origin Location",
                "Body Mass"
            ]
        )

        st.dataframe(input_df, use_container_width=True)
