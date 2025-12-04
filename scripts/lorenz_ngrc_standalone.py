import torch
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import itertools
from math import factorial
import time
import optuna

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
dtype = torch.float64

class NGRC:
    """
    A Next Generation Reservoir Computer model, adopted from Gauthier et al. (https://arxiv.org/abs/2106.07688), implemented with torch.
    """
    def __init__(self, k, s, poly_degree, alpha):
        self.k = k
        self.s = s
        self.poly_degree = poly_degree
        self.W_out = None
        self.alpha = alpha

    @staticmethod
    def _n_choose_k_with_replacement(n, k):
        """Calculates combinations with replacement, used for polynomial feature counting."""
        if n <= 0 and k > 0:
            return 0
        if n == 0 and k == 0:
            return 1
        return factorial(n + k - 1) // (factorial(k) * factorial(n - 1))

    def get_feature_count(self, n_dims):
        """Calculates the total number of polynomial features based on model config."""
        if n_dims <= 0:
            return 0
        d_lin = self.k * n_dims
        # Bias + linear terms
        n_features = 1 + d_lin
        if self.poly_degree >= 2:
            n_features += self._n_choose_k_with_replacement(d_lin, 2)
        if self.poly_degree >= 3:
            # The optuna search doesn't have 3, but it works
            n_features += self._n_choose_k_with_replacement(d_lin, 3)
        return n_features

    def get_required_history_len(self):
        """Calculates the number of past data points needed to make one prediction."""
        return (self.k - 1) * self.s + 1

    def predict(self, initial_history, n_steps, stability_threshold=1e6):
        """
        Generates future predictions step-by-step (autonomously).
        """
        if self.W_out is None:
            raise RuntimeError("Model not trained")

        input_dim = initial_history.shape[1]
        current_history = initial_history.clone()
        predictions = []

        for _ in range(n_steps):
            omega_t = _construct_feature_matrix(self, current_history)

            # Predict the change in X and update
            delta_X_pred = self.W_out @ omega_t
            X_pred = current_history[-1] + delta_X_pred.squeeze(1)

            # Check for numerical instability
            if torch.any(torch.abs(X_pred) > stability_threshold):
                print("Prediction unstable, stopping.")
                break

            predictions.append(X_pred)
            # Update history: drop the oldest point, add the newest prediction
            current_history = torch.cat((current_history[1:], X_pred.unsqueeze(0)), dim=0)

        if not predictions:
            return torch.empty((0, input_dim), device=device, dtype=dtype)

        return torch.stack(predictions)


    def train(self, ngrc_model, train_data, verbose=True):
        """Trains the NGRC model using Ridge Regression."""
        if verbose:
            print(f"--- Starting Training for Polynomial model ---")

        n_features = ngrc_model.get_feature_count(train_data.shape[1])
        if verbose:
            print(f"Model configured with {n_features} features.")

        omega, Yd = prepare_matrices_for_training(ngrc_model, train_data)

        I = torch.eye(omega.shape[0], device=device, dtype=dtype)
        # Solve (Omega * Omega^T + alpha*I) * W^T = Yd * Omega^T
        A = omega @ omega.T + self.alpha * I
        B = Yd @ omega.T

        try:
            # Use fast and stable Cholesky decomposition
            L, info = torch.linalg.cholesky_ex(A)
            if info > 0:
                raise torch.linalg.LinAlgError("Matrix is not positive-definite")
            W_T = torch.cholesky_solve(B.T, L)
            ngrc_model.W_out = W_T.T
        except torch.linalg.LinAlgError:
            # Fallback to pseudoinverse if Cholesky fails
            if verbose:
                print("Cholesky decomposition failed. Using slower pseudoinverse.")
            A_pinv = torch.linalg.pinv(A)
            ngrc_model.W_out = B @ A_pinv

        if verbose:
            print("--- Training complete ---")

        return ngrc_model


def _construct_feature_matrix(ngrc_model, data):
    """
    Constructs the polynomial feature matrix Omega from the input data.
    """
    k, s = ngrc_model.k, ngrc_model.s
    history_len = (k - 1) * s
    N_points, N_dims = data.shape

    if N_points <= history_len:
        return torch.empty((ngrc_model.get_feature_count(N_dims), 0), device=device, dtype=dtype)

    # Create the linear part of the features
    O_lin = torch.cat([data[history_len - i * s: N_points - i * s] for i in range(k)], dim=1)
    n_usable_points, num_lin_features = O_lin.shape

    bias = torch.ones(n_usable_points, 1, device=device, dtype=dtype)

    poly_terms = [bias, O_lin]
    if ngrc_model.poly_degree >= 2:
        # Combinations with replacement
        indices_i, indices_j = torch.triu_indices(num_lin_features, num_lin_features, offset=0, device=device)
        poly_terms.append(O_lin[:, indices_i] * O_lin[:, indices_j])
    if ngrc_model.poly_degree >= 3:
        indices_iter = itertools.combinations_with_replacement(range(num_lin_features), 3)
        indices_tensor = torch.tensor(list(indices_iter), device=device, dtype=torch.long)
        poly_terms.append(O_lin[:, indices_tensor[:, 0]] * O_lin[:, indices_tensor[:, 1]] * O_lin[:, indices_tensor[:, 2]])

    return torch.cat(poly_terms, dim=1).T


def prepare_matrices_for_training(ngrc_model, train_data):
    """Prepares the Omega and Yd matrices for Ridge Regression."""
    history_len = ngrc_model.get_required_history_len() - 1
    omega = _construct_feature_matrix(ngrc_model, train_data[:-1])
     # Yd is the one-step-ahead difference
    Yd = (train_data[history_len + 1:] - train_data[history_len:-1]).T

    # Make sure matrices have the same number of samples
    num_samples = Yd.shape[1]
    omega = omega[:, :num_samples]

    return omega, Yd


def objective(trial, train_data, validation_data):
    """
    Optuna objective function; defines one trial for hyperparameter search.
    """
    # parameter suggestions
    model_params = {
        'k': trial.suggest_int('k', 2, 8),
        's': trial.suggest_int('s', 1, 3),
        # Restrict to degree 2 to ensure very fast runtime, but it also works for 3
        'poly_degree': trial.suggest_int('poly_degree', 2, 2),
        'alpha': trial.suggest_float('alpha', 1e-9, 1e-3, log=True)
    }


    try:
        model = NGRC(**model_params)

        required_history = model.get_required_history_len()
        if required_history >= len(train_data):
            return float('inf') # Prune invalid trials

        model.train(model, train_data, verbose=False)

        initial_history = train_data[-required_history:]
        validation_forecast_steps = len(validation_data)
        predicted_vals = model.predict(initial_history, n_steps=validation_forecast_steps)

        dt = 0.01
        vft = calculate_valid_forecast_time(predicted_vals, validation_data, dt)

        return vft # Optuna minimizes

    except (torch.linalg.LinAlgError, RuntimeError) as e:
        print(f"Trial failed with error: {e}")
        return float('inf')

def calculate_valid_forecast_time(predicted_data, true_data, dt, error_threshold=0.5):
    """
    Calculates the Valid Forecast Time (VFT) in time units.
    """
    if predicted_data.shape[0] == 0:
        return 0.0

    min_len = min(predicted_data.shape[0], true_data.shape[0])
    predicted_data = predicted_data[:min_len]
    true_data = true_data[:min_len]

    # Normalize error threshold (by the standard deviation of the true signal's variance)
    true_std = torch.std(true_data)
    if true_std < 1e-9:
        return 0.0 # Avoid division by zero error if data is too flat

    instantaneous_error = torch.sqrt(torch.mean((predicted_data - true_data)**2, dim=1))
    normalized_error = instantaneous_error / true_std

    # Finds the first time step where error exceeds the threshold
    invalid_indices = torch.where(normalized_error > error_threshold)[0]

    first_invalid_step = invalid_indices[0].item() if len(invalid_indices) > 0 else min_len

    return first_invalid_step * dt




if __name__ == "__main__":
    # Data generation
    def lorenz_system(t, xyz, sigma=10, rho=28, beta=8/3):
        x, y, z = xyz
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    sol = solve_ivp(lorenz_system, (0, 400), [0, 1, 1.05], t_eval=np.arange(0, 400, 0.01), rtol=1e-9, atol=1e-9)
    # Ignore the transient
    data = torch.tensor(sol.y.T[2000:], dtype=dtype, device=device)
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

        print(f"Best trial found: VFT={-study.best_value:.4f}")
        print("Best hyperparameters:", best_params)

        final_model = NGRC(**best_params)

        # Train on combined training and validation data for the best possible final model
        full_train_data = torch.cat([train_data, validation_data], dim=0)
        final_model.train(final_model, full_train_data)

        # seed the forecast with the end of the final full train data
        initial_history_len = final_model.get_required_history_len()
        initial_history = full_train_data[-initial_history_len:]
        predicted_data = final_model.predict(initial_history, n_steps=len(test_data)).cpu().numpy()
        true_data_to_plot = test_data.cpu().numpy()

        # Plotting
        print("Plotting final comparison...")
        fig = plt.figure(figsize=(15, 6))
        fig.suptitle("Best Polynomial Model Performance", fontsize=16)

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(true_data_to_plot[:, 0], true_data_to_plot[:, 1], true_data_to_plot[:, 2], lw=0.5, color='k', label="True Attractor")
        ax1.plot(predicted_data[:, 0], predicted_data[:, 1], predicted_data[:, 2], lw=0.7, ls='-', color='r', label="Polynomial Pred.")
        ax1.set_title("Attractor Reconstruction")
        ax1.legend()

        ax2 = fig.add_subplot(122)
        plot_len = min(len(true_data_to_plot), len(predicted_data), 4000)
        time_axis = np.arange(plot_len) * 0.01
        ax2.plot(time_axis, true_data_to_plot[:plot_len, 0], color='k', label="True X(t)")
        ax2.plot(time_axis, predicted_data[:plot_len, 0], linestyle='--', color='r', label="Polynomial Pred. X(t)")
        ax2.set_title("Short-term Forecast")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    else:
        print("\nOptuna study did not succeed. ")