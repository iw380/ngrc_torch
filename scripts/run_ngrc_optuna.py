import sys
import os

import torch
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import optuna
from ngrc_torch.model import NGRC, NGRCTrainer
from ngrc_torch.utils.metrics import calculate_valid_forecast_time
from ngrc_torch.data.lorenz import generate_lorenz_data
from ngrc_torch.utils.plotting import plot_predictions
from ngrc_torch.config import DEVICE, DTYPE

device = DEVICE
print(f"Using device: {device}")
dtype = DTYPE

def objective(trial, train_data, validation_data):
    """
    Optuna objective function; defines one trial for hyperparameter search.
    """
    # parameter suggestions
    model_params = {
        'k': trial.suggest_int('k', 2, 8),
        's': trial.suggest_int('s', 1, 3),
        # Restrict to degree 2 to ensure very fast runtime, but it also works for 3
        'poly_degree': trial.suggest_int('poly_degree', 2, 2)
    }
    alpha = trial.suggest_float('alpha', 1e-9, 1e-3, log=True)

    try:
        model = NGRC(**model_params)
        trainer = NGRCTrainer(alpha=alpha)
        
        required_history = model.get_required_history_len()
        if required_history >= len(train_data):
            return float('inf') # Prune invalid trials

        trainer.train(model, train_data, verbose=False)

        initial_history = train_data[-required_history:]
        validation_forecast_steps = len(validation_data)
        predicted_vals = model.predict(initial_history, n_steps=validation_forecast_steps)

        dt = 0.01 
        vft = calculate_valid_forecast_time(predicted_vals, validation_data, dt)
        
        return vft # Optuna minimizes

    except (torch.linalg.LinAlgError, RuntimeError) as e:
        print(f"Trial failed with error: {e}")
        return float('inf')

if __name__ == "__main__":
    # Data generation
    lorenz_df = generate_lorenz_data(duration=400.0, dt=0.01, filepath=None)
    # generate_lorenz_data already removes transient
    data = torch.tensor(lorenz_df.values, dtype=dtype, device=device)
    print(f"Generated Lorenz data with shape: {data.shape}")

    # Data splitting
    train_end = int(len(data) * 0.5)
    val_end = train_end + int(len(data) * 0.25)
    train_data = data[:train_end]
    validation_data = data[train_end:val_end]
    test_data = data[val_end:]
    print(f"Data split: Train({len(train_data)}), Validation({len(validation_data)}), Test({len(test_data)})")

    # Hyperparameter search
    print("STARTING OPTUNA HYPERPARAMETER SEARCH")
    study = optuna.create_study(direction='maximize') # Maximize VFT
    study.optimize(lambda trial: objective(trial, train_data, validation_data), n_trials=50)
    
    # Evaluation -------
    if study.best_trial.state == optuna.trial.TrialState.COMPLETE:
        print("\n--- Evaluating best model on unseen test set... ---")

        best_params = study.best_trial.params.copy()
        final_alpha = best_params.pop('alpha')
        
        print(f"Best trial found: VFT={-study.best_value:.4f}")
        print("Best hyperparameters:", best_params)

        final_model = NGRC(**best_params)

        # Train on combined training and validation data for the best possible final model
        full_train_data = torch.cat([train_data, validation_data], dim=0)
        final_trainer = NGRCTrainer(alpha=final_alpha)
        final_trainer.train(final_model, full_train_data)

        # seed the forecast with the end of the final full train data
        initial_history_len = final_model.get_required_history_len()
        initial_history = full_train_data[-initial_history_len:]
        predicted_data = final_model.predict(initial_history, n_steps=len(test_data)).cpu().numpy()
        true_data_to_plot = test_data.cpu().numpy()

        # Plotting
        print("Plotting final comparison...")
        predictions = {"Polynomial Pred.": predicted_data}
        plot_predictions(true_data_to_plot, predictions, dt=0.01, title="Best Polynomial Model Performance")
    else:
        print("\nOptuna study did not succeed. ")
