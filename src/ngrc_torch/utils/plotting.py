import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_predictions(
    true_data,
    predictions,
    dt=0.01,
    title="Model Comparison",
    plot_dims=(0, 1, 2),
    time_series_dim=0
):
    """
    Visualizes one or more model predictions against the ground truth.

    Args:
        true_data (np.ndarray or pd.DataFrame): The ground truth data.
        predictions (dict): A dictionary where keys are model names (str) and
                            values are their prediction arrays (np.ndarray).
        dt (float): The time step of the data.
        title (str): The main title for the plot figure.
        plot_dims (tuple): The three indices of the dimensions to use for the 3D plot.
        time_series_dim (int): The index of the dimension to plot over time.
    """
    if isinstance(true_data, pd.DataFrame): true_data = true_data.values

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(title, fontsize=16)

    # Attractor Plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(true_data[:, plot_dims[0]], true_data[:, plot_dims[1]], true_data[:, plot_dims[2]],
             lw=0.5, color='k', label="True Attractor")

    # Time Series Plot
    ax2 = fig.add_subplot(1, 2, 2)
    plot_len = min(len(true_data), 4000)
    time_axis = np.arange(plot_len) * dt
    ax2.plot(time_axis, true_data[:plot_len, time_series_dim], color='k', label="True Signal")

    # Plot each prediction
    for model_name, pred_data in predictions.items():
        if pred_data is None or len(pred_data) == 0:
            print(f"Skipping empty prediction for model: '{model_name}'")
            continue

        # Plot on 3D attractor
        ax1.plot(pred_data[:, plot_dims[0]], pred_data[:, plot_dims[1]], pred_data[:, plot_dims[2]],
                 lw=0.8, ls='--', label=f"{model_name} Pred.")

        # Plot on time series
        pred_plot_len = min(len(pred_data), plot_len)
        pred_time_axis = np.arange(pred_plot_len) * dt
        ax2.plot(pred_time_axis, pred_data[:pred_plot_len, time_series_dim],
                 ls='--', label=f"{model_name} Pred.")

    ax1.set_title("Attractor Reconstruction")
    ax1.set_xlabel(f"Dim {plot_dims[0]}"); ax1.set_ylabel(f"Dim {plot_dims[1]}"); ax1.set_zlabel(f"Dim {plot_dims[2]}")
    ax1.legend()

    ax2.set_title(f"Time Series (Dimension {time_series_dim})")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel(f"Value")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
