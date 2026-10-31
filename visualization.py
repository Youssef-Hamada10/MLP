import numpy as np
import matplotlib.pyplot as plt

from preprocessing import FEATURES, ORIGIN_LOCATIONS

def plot_mlp_decision_boundary(
    model,
    training_results,
    feature1,
    feature2,
    scale="Standardized",
    resolution=300
):
    # Get feature indices
    feature1_idx = FEATURES.index(feature1)
    feature2_idx = FEATURES.index(feature2)

    if feature1_idx == feature2_idx:
        raise ValueError("Select two different features.")

    # Get data
    X_train = training_results["X_train_scaled"]
    X_test = training_results["X_test_scaled"]

    y_train = training_results["y_train"]
    y_test = training_results["y_test"]

    mean = training_results["mean"]
    std = training_results["std"]

    # Select the requested scale
    if scale == "Standardized":
        X_tr = X_train
        X_te = X_test
        x_label = f"{feature1} (Standardized)"
        y_label = f"{feature2} (Standardized)"

    elif scale == "Original":
        X_tr = training_results["X_train"]
        X_te = training_results["X_test"]
        x_label = feature1
        y_label = feature2

    else:
        raise ValueError("Scale must be Standardized or Original.")

    # Get plotting limits
    all_x = np.concatenate([
        X_tr[:, feature1_idx],
        X_te[:, feature1_idx]
    ])

    all_y = np.concatenate([
        X_tr[:, feature2_idx],
        X_te[:, feature2_idx]
    ])

    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    x_margin = (x_max - x_min) * 0.08 or 1
    y_margin = (y_max - y_min) * 0.08 or 1

    x_min -= x_margin
    x_max += x_margin
    y_min -= y_margin
    y_max += y_margin

    # Create grid
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Fix remaining features at their training means
    base_point = mean.copy()

    # Insert grid values into the selected features
    grid = np.tile(base_point, (xx.size, 1))

    grid[:, feature1_idx] = xx.ravel()
    grid[:, feature2_idx] = yy.ravel()

    # Convert grid to standardized values if needed
    if scale == "Standardized":
        grid_scaled = (grid - mean) / std
    else:
        grid_scaled = grid

    # Predict all grid points
    Z = model.predict(grid_scaled)
    Z = Z.reshape(xx.shape)

    # Plot decision regions
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.contourf(
        xx,
        yy,
        Z,
        levels=[-0.5, 0.5, 1.5, 2.5],
        alpha=0.25,
        cmap="viridis"
    )

    ax.contour(
        xx,
        yy,
        Z,
        levels=[0.5, 1.5],
        colors="black",
        linewidths=1.5
    )

    # Plot training and testing samples
    species = ["Adelie", "Chinstrap", "Gentoo"]
    markers = ["o", "s", "^"]
    colors = ["blue", "orange", "green"]

    for cls in range(3):

        train_mask = y_train == cls
        test_mask = y_test == cls

        ax.scatter(
            X_tr[train_mask, feature1_idx],
            X_tr[train_mask, feature2_idx],
            color=colors[cls],
            marker=markers[cls],
            s=45,
            alpha=0.75,
            label=f"{species[cls]} (Train)"
        )

        ax.scatter(
            X_te[test_mask, feature1_idx],
            X_te[test_mask, feature2_idx],
            facecolors="none",
            edgecolors=colors[cls],
            marker=markers[cls],
            s=75,
            linewidths=1.5,
            label=f"{species[cls]} (Test)"
        )

    # Axis labels
    ax.set_xlabel(x_label, fontsize=11, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=11, fontweight="bold")

    # OriginLocation labels
    if scale == "Original" and feature1 == "OriginLocation":
        ax.set_xticks(
            list(ORIGIN_LOCATIONS.values()),
            list(ORIGIN_LOCATIONS.keys())
        )

    if scale == "Original" and feature2 == "OriginLocation":
        ax.set_yticks(
            list(ORIGIN_LOCATIONS.values()),
            list(ORIGIN_LOCATIONS.keys())
        )

    ax.set_title(
        f"MLP Decision Regions: {feature1} vs {feature2}",
        fontsize=13,
        fontweight="bold"
    )

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()

    return fig
