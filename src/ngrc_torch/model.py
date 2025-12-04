import torch
import numpy as np
import itertools
from math import factorial
from .config import DEVICE, DTYPE

device = DEVICE
dtype = DTYPE

class NGRC:
    """
    A Next Generation Reservoir Computer model, adopted from Gauthier et al. (https://arxiv.org/abs/2106.07688), implemented with torch. 
    """
    def __init__(self, k, s, poly_degree):
        self.k = k
        self.s = s
        self.poly_degree = poly_degree
        self.W_out = None

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

class NGRCTrainer:
    """Trains an NGRC model"""
    def __init__(self, alpha=1e-5):
        self.alpha = alpha

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

