import numpy as np
import pandas as pd

def evaluate_prediction(
    true_data,
    pred_data,
    dt=0.01,
    short_term_window_steps=500,
    vpt_threshold_ratio=0.4
):
    """
    Evaluates a time-series prediction against the ground truth.
    """
    if isinstance(true_data, pd.DataFrame): true_data = true_data.values
    if isinstance(pred_data, pd.DataFrame): pred_data = pred_data.values
    n_pred_steps = pred_data.shape[0]; n_true_steps = true_data.shape[0]
    is_stable = n_pred_steps >= n_true_steps
    eval_len = min(n_pred_steps, short_term_window_steps)
    short_term_mse = np.mean((true_data[:eval_len] - pred_data[:eval_len])**2)
    attractor_magnitude = np.linalg.norm(true_data, axis=1)
    divergence_threshold = np.std(attractor_magnitude) * vpt_threshold_ratio
    error_distance = np.linalg.norm(true_data[:n_pred_steps] - pred_data, axis=1)
    divergence_indices = np.where(error_distance > divergence_threshold)[0]
    vpt_steps = divergence_indices[0] if len(divergence_indices) > 0 else n_pred_steps
    valid_prediction_time = vpt_steps * dt
    results = {
        "is_stable": is_stable, "predicted_steps": n_pred_steps, "total_true_steps": n_true_steps,
        "short_term_mse": short_term_mse, "valid_prediction_time": valid_prediction_time,
        "vpt_divergence_threshold": divergence_threshold
    }
    return results

def calculate_valid_forecast_time(predicted_data, true_data, dt, error_threshold=0.5):
    """
    Calculates the Valid Forecast Time (VFT) in time units.
    """
    if isinstance(predicted_data, pd.DataFrame): predicted_data = predicted_data.values
    if isinstance(true_data, pd.DataFrame): true_data = true_data.values
    
    # Ensure numpy arrays
    if hasattr(predicted_data, 'cpu'): predicted_data = predicted_data.cpu().numpy()
    if hasattr(true_data, 'cpu'): true_data = true_data.cpu().numpy()

    if predicted_data.shape[0] == 0:
        return 0.0

    min_len = min(predicted_data.shape[0], true_data.shape[0])
    predicted_data = predicted_data[:min_len]
    true_data = true_data[:min_len]

    # Normalize error threshold (by the standard deviation of the true signal's variance)
    true_std = np.std(true_data)
    if true_std < 1e-9:
        return 0.0 # Avoid division by zero error if data is too flat

    instantaneous_error = np.sqrt(np.mean((predicted_data - true_data)**2, axis=1))
    normalized_error = instantaneous_error / true_std
    
    # Finds the first time step where error exceeds the threshold
    invalid_indices = np.where(normalized_error > error_threshold)[0]
    
    first_invalid_step = invalid_indices[0] if len(invalid_indices) > 0 else min_len
        
    return first_invalid_step * dt
