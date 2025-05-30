import numpy as np
from typing import Tuple
import itertools


def all_states(n: int) -> np.ndarray:
    """
    Generates a matrix of shape (2^n, n) with all possible spin configurations in {-1, 1}.
    """
    # Each state is a tuple in {-1, 1}^n
    states = np.array(list(itertools.product([-1, 1], repeat=n)))
    return states


def compute_model_statistics(
    J: np.ndarray, h: np.ndarray, states: np.ndarray, beta: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given current J and h, computes the exact model expectations:
      model_s = <s> and model_s_s = <s s^T>
    using all states in 'states'.
    """
    # Each row of states is a vector s of size n_features
    # Compute the energy (in the exponent) for each state:
    # Note: We use the definition: E(s) = -[ s·h + s^T J s ]
    # so the weight is: exp(beta*(s·h + s^T J s))
    energies = -(np.dot(states, h) + 0.5 * np.sum(states * (states @ J), axis=1))
    # Boltzmann weight:
    weights = np.exp(-beta * energies)
    Z = np.sum(weights)

    # Expectation of s:
    model_s = (states.T @ weights) / Z
    # Expectation of s_i s_j:
    # Weighted: each state contributes with outer(s, s)
    model_s_s = (states.T @ (states * weights[:, np.newaxis])) / Z

    return model_s, model_s_s


def gradient_descent_ising(
    data: np.ndarray,
    learning_rate: float,
    epochs: int,
    reg_strength: float = 0.0,
    beta: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Infers J_ij (interaction matrix) and h_i (local biases) using exact gradient descent.

    Parameters:
        data (np.ndarray): Binary array of shape (n_samples, n_features) with values in {-1, 1}.
        learning_rate (float): Learning rate.
        epochs (int): Number of iterations.
        reg_strength (float): Regularization strength (you can set 0 for small systems).
        beta (float): Inverse temperature.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Inferred interaction matrix J and bias vector h.
    """
    n_samples, n_features = data.shape

    # Parameter initialization: symmetric J without diagonal and h
    J = np.random.normal(0, 0.1, size=(n_features, n_features))
    h = np.random.normal(0, 0.1, size=n_features)
    np.fill_diagonal(J, 0)

    # Precompute all possible states (2^n_features)
    states = all_states(n_features)  # Shape: (2^n_features, n_features)

    # Compute statistics from data
    avg_s = np.mean(data, axis=0)
    avg_s_s = (data.T @ data) / n_samples

    for epoch in range(epochs):
        # Compute exact model expectations with current parameters
        model_s, model_s_s = compute_model_statistics(J, h, states, beta)

        # Gradients (difference between data and model statistics)
        dh = avg_s - model_s - reg_strength * h
        dJ = avg_s_s - model_s_s - reg_strength * J

        # Parameter update
        h += learning_rate * dh
        J += learning_rate * dJ

        # Enforce symmetry of J and zero diagonal
        J = (J + J.T) / 2
        np.fill_diagonal(J, 0)

        # Show progress every 10 epochs
        if epoch % 10 == 0:
            max_dh = np.max(np.abs(dh))
            max_dJ = np.max(np.abs(dJ))
            print(f"Epoch {epoch}, Max |dh|: {max_dh:.4f}, Max |dJ|: {max_dJ:.4f}")

    return J, h
