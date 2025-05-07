import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import math
import random
import seaborn as sns
import scipy.sparse as sp
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm
import torch
from sklearn.model_selection import KFold

import torch.multiprocessing as mp
from itertools import product
from matplotlib.collections import LineCollection

import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
from ocean_flow_parallel import (
    compute_log_likelihood,
    rbf_kernel,
    compute_conditional,
    compute_ll_for_var_lengthscale_of_U_and_V_component,
    predict_conditional_mean_variance_fixed_hyperparams,
    find_future_magnitude_given_coordinates_in_kilometers,
    simulate_expanded_particle_debris_scattering,
)


# Global Variables
DATA_DIR = "OceanFlow/"
NUM_TIMESTEPS = 100  # T = 1 to 100
GRID_SPACING_KM = 3
TIME_SPACING_H = 3
PAUSE_BETWEEN_FRAMES = 100  # milliseconds


def squared_exponential_kernel(x1, x2, sigma, lengthscale):
    """
    Compute the squared exponential kernel between two tensors using PyTorch.

    Parameters:
    - x1: Tensor of shape (n, d)
    - x2: Tensor of shape (m, d)
    - sigma: scalar or 0-D tensor
    - lengthscale: scalar or 0-D tensor

    Returns:
    - kernel: Tensor of shape (n, m)
    """
    # Ensure inputs are 2D
    if x1.dim() == 1:
        x1 = x1.unsqueeze(1)
    if x2.dim() == 1:
        x2 = x2.unsqueeze(1)

    # Compute squared Euclidean distance
    diff = x1.unsqueeze(1) - x2.unsqueeze(0)  # shape: (n, m, d)
    squared_dist = torch.sum(diff**2, dim=-1)  # shape: (n, m)

    return sigma**2 * torch.exp(-squared_dist / (2 * lengthscale**2))


def compute_log_likelihood(x_test, mu_pred, cov_pred, tau):
    """This function will compute the log-likelihood using the
    conditional mean, variance, and test data.

    Parameters
    ----------
    x_test : np.array
        These are the test values.
    mu_pred : np.array
        This is the predicted mean.
    cov_pred : np.array
        This is the predicted covariance.
    tau : float
        This is the scalar representing noise.
    """
    ll = (
        -0.5
        * (x_test - mu_pred).T
        @ torch.linalg.inv(cov_pred + tau * torch.eye(cov_pred.shape[0]).to(device))
        @ (x_test - mu_pred)
        - 0.5
        * torch.log(
            torch.linalg.det(cov_pred + tau * torch.eye(cov_pred.shape[0]).to(device))
        )
        - x_test.shape[0] / 2 * torch.log(torch.tensor(2 * torch.pi))
    )
    return ll


def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))


def select_distant_indices(coords, num_points=10, distance_percent=0.8):
    """Select `num_points` indices such that each is `distance_percent` away from the others."""
    n = len(coords)
    max_possible_distance = euclidean_distance((0, 0), (504, 555))
    required_min_distance = distance_percent * max_possible_distance

    # Pick first point randomly
    selected_indices = [random.randint(0, n - 1)]

    while len(selected_indices) < num_points:
        candidates = list(set(range(n)) - set(selected_indices))
        random.shuffle(candidates)
        for idx in tqdm(candidates, desc="Selecting Indices...", ascii="░▒▓█"):
            if all(
                euclidean_distance(coords[idx], coords[other]) >= required_min_distance
                for other in selected_indices
            ):
                selected_indices.append(idx)
                break
        else:
            # If we cannot find a candidate (too strict requirement), relax the minimum distance slightly
            required_min_distance *= 0.95

    return selected_indices


def compute_magnitude(u, v):
    return np.sqrt(u**2 + v**2)


def load_flow_data(timestep):
    u_path = os.path.join(DATA_DIR, f"{timestep}u.csv")
    v_path = os.path.join(DATA_DIR, f"{timestep}v.csv")
    u = np.loadtxt(u_path, delimiter=",")
    v = np.loadtxt(v_path, delimiter=",")
    return u, v


def plot_vector_field(u, v, ax, title=""):
    ax.clear()
    y, x = np.mgrid[0 : u.shape[0], 0 : u.shape[1]]
    q = ax.quiver(
        x * GRID_SPACING_KM,
        y * GRID_SPACING_KM,
        u,
        v,
        compute_magnitude(u, v),
        scale=50,
        cmap="viridis",
    )

    # Add a colorbar to show magnitude scale
    cbar = plt.colorbar(q, ax=ax)
    cbar.set_label("Velocity Magnitude (km/H)")  # Customize unit if known
    ax.set_title(title)
    ax.set_xlabel("Distance X (km)")
    ax.set_ylabel("Distance Y (km)")
    ax.set_aspect("equal")


def plot_single_frame(timestep):
    u, v = load_flow_data(timestep)
    fig, ax = plt.subplots()
    plot_vector_field(u, v, ax, f"Flow Field at Time Step {timestep}")
    plt.show()


def compute_conditional_mean_variance_fixed_hyperparams(
    U: torch.Tensor,
    sigma: float,
    ell: float,
    tau: float,
    x_coord: int,
    y_coord: int,
    device: str = "cuda",
):
    """
    Compute conditional GP mean and variance at fixed pixel (x_coord, y_coord) over time using full PyTorch ops.

    Parameters:
        U       : torch.Tensor of shape (T, H, W), velocity field
        sigma   : float, GP signal variance
        ell     : float, GP lengthscale
        tau     : float, GP noise standard deviation
        x_coord : int, x index of pixel
        y_coord : int, y index of pixel
        device  : str, "cuda" or "cpu"

    Returns:
        mean_star : torch.Tensor of shape (T,)
        var_star  : torch.Tensor of shape (T,)
    """
    y_train = U[:, y_coord, x_coord].to(torch.float64).to(device)  # shape (T,)
    T = y_train.shape[0]
    X_train = torch.arange(T, dtype=torch.float64, device=device).reshape(-1, 1)

    K = rbf_kernel(X_train, X_train, sigma, ell)
    K += (tau**2) * torch.eye(T, dtype=torch.float64, device=device)

    # Inverse can be unstable. Use Cholesky if needed for numerical stability
    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(y_train.unsqueeze(1), L)  # shape (T, 1)

    mean_star = (K @ alpha).squeeze()  # shape (T,)
    var_star = torch.diag(K - K @ torch.cholesky_solve(K, L))  # shape (T,)

    return mean_star, var_star


def causal_moving_average(data, window_size):
    # Causal moving average: only use past and current values to compute the average
    return torch.tensor(
        [data[max(i - window_size + 1, 0) : i + 1].mean() for i in range(len(data))],
        device=data.device,
        dtype=torch.float64,
    )


def predict_conditional_mean_variance_fixed_hyperparams(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    y_train: torch.Tensor,
    sigma: float,
    ell: float,
    tau: float,
):
    """
    Predict the conditional mean and variance using a Gaussian Process
    with fixed hyperparameters.

    Parameters:
    - X_train (torch.Tensor): shape (n_train, 1)
    - X_test (torch.Tensor): shape (n_test, 1)
    - y_train (torch.Tensor): shape (n_train, 1)
    - sigma (float): signal variance (amplitude)
    - ell (float): lengthscale
    - tau (float): observation noise std deviation

    Returns:
    - mean_star (torch.Tensor): predictive mean, shape (n_test, 1)
    - var_star (torch.Tensor): predictive variance (diagonal), shape (n_test,)
    """

    device = X_train.device

    # Compute kernel matrices using PyTorch
    K_train = rbf_kernel(X_train, X_train, sigma, ell) + (tau**2) * torch.eye(
        X_train.shape[0], device=device
    )

    K_s = rbf_kernel(X_test, X_train, sigma, ell)
    K_ss = rbf_kernel(X_test, X_test, sigma, ell)

    # Solve using Cholesky decomposition for stability (optional)
    L = torch.linalg.cholesky(K_train)
    alpha = torch.cholesky_solve(y_train, L)

    # Predictive mean
    mean_star = K_s @ alpha

    # Predictive variance (diagonal)
    v = torch.cholesky_solve(K_s.T, L)
    var_star = torch.diag(K_ss - K_s @ v)

    return mean_star, var_star


def plot_particle_coordinates_traces_and_trajectories_for_timestep(
    x_particle_coordinates, y_particle_coordinates, MAG, TIMESTEP=0, ARROW_SCALE=100
):
    """This function plots the particle positions, trace of past positions, and trajectory of future positions.

    Parameters
    ----------
    TIMESTEP : int
        This is the timestep (time index) for which to plot the data.
    ARROW_SCALE : int
        This is the factor to scale the trajectory arrows by for visibility purposes.
    x_particle_coordinates : torch.tensor (n_particles, n_timesteps)
        These are the x coordinates of the particle.
    y_particle_coordinates : torch.tensor (n_particles, n_timesteps)
        These are the y coordinates of each particle.
    """

    num_particles = x_particle_coordinates.shape[0]

    cmap = plt.get_cmap(
        "tab20"
    )  # tab20 has 20 distinct colors; better for categorical data
    colors = [
        cmap(i % 20) for i in range(num_particles)
    ]  # wrap around if >20 particles

    # Create a mapping: particle index -> color
    particle_colors = {i: colors[i] for i in range(x_particle_coordinates.shape[0])}

    fig, ax = plt.subplots(figsize=(12, 8))

    # Display the grid

    MAG_CPU = MAG.cpu()
    U_CPU = U.cpu()
    V_CPU = V.cpu()

    y_particle_coordinates_cpu = y_particle_coordinates.cpu()
    x_particle_coordinates_cpu = x_particle_coordinates.cpu()

    cax = ax.imshow(
        MAG_CPU[TIMESTEP, :, :],
        origin="lower",  # (0, 0) at the bottom left
        cmap="viridis",  # colormap
        extent=[
            0,
            MAG_CPU[TIMESTEP, :, :].shape[1] * GRID_SPACING_KM,
            0,
            MAG_CPU[TIMESTEP, :, :].shape[0] * GRID_SPACING_KM,
        ],  # [x_min, x_max, y_min, y_max]
        aspect="equal",
    )  # square pixel aspect ratio

    # Add a colorbar
    fig.colorbar(cax, ax=ax, label="Magnitude of Flow (Km/H)")

    # Label axes
    ax.set_xlabel("X coordinate (Km)")
    ax.set_ylabel("Y coordinate (Km)")
    ax.set_title(
        f"Simulation of particle movement superimposed onto \nthe magnitude of flow in the Phillipine Archipellago\n at Hour {(TIMESTEP) * 3} to {(TIMESTEP+1) * 3}"
    )

    # Plot the particle

    for particle in range(10):
        # Convert Coordinate positions from index to Km
        x = x_particle_coordinates_cpu[particle][TIMESTEP]
        y = y_particle_coordinates_cpu[particle][TIMESTEP]

        # Plot Position
        ax.scatter(
            x,
            y,
            color=particle_colors[particle],
            s=20,
            label=f"Particle {particle + 1}",
        )

        # Plot Current Vector
        ax.arrow(
            x,
            y,
            ARROW_SCALE
            * U_CPU[
                TIMESTEP,
                (y_particle_coordinates_cpu[particle][TIMESTEP] / 3).int(),
                (x_particle_coordinates_cpu[particle][TIMESTEP] / 3).int(),
            ],
            ARROW_SCALE
            * V_CPU[
                TIMESTEP,
                (y_particle_coordinates_cpu[particle][TIMESTEP] / 3).int(),
                (x_particle_coordinates_cpu[particle][TIMESTEP] / 3).int(),
            ],
            head_width=20,
            head_length=20,
            fc=colors[particle],
            ec=colors[particle],
        )

        # Plot particle trace
        # Plot Position
        if TIMESTEP > 0:
            for past_index in range(0, TIMESTEP):
                x_past = x_particle_coordinates_cpu[particle][past_index]
                y_past = y_particle_coordinates_cpu[particle][past_index]

                ax.scatter(
                    x_past, y_past, color=particle_colors[particle], alpha=0.1, s=1
                )

    fig.text(
        0.1,
        0.02,
        "* Trajectory vectors are scaled by 2 orders of magnitude for visibility",
        fontsize=10,
        ha="left",
    )

    # Add legend
    ax.legend()

    plt.show()


def plot_particle_coordinates_traces_and_trajectories_for_timestep(
    x_particle_coordinates,
    y_particle_coordinates,
    MAG,
    U,
    V,
    TIMESTEP=0,
    ARROW_SCALE=100,
):
    """This function plots the particle positions, trace of past positions, and trajectory of future positions.

    Parameters
    ----------
    TIMESTEP : int
        This is the timestep (time index) for which to plot the data.
    ARROW_SCALE : int
        This is the factor to scale the trajectory arrows by for visibility purposes.
    x_particle_coordinates : torch.tensor (n_particles, n_timesteps)
        These are the x coordinates of the particle.
    y_particle_coordinates : torch.tensor (n_particles, n_timesteps)
        These are the y coordinates of each particle.
    MAG: torch.float64
        This is the magnitude dataframe of the combined flows of the
        U and V matrices.
    U: torch.float64
        This is the U component (horizontal) flow.
    V: torch.float64
        This is the V component (vertical) flow.
    """

    num_particles = x_particle_coordinates.shape[0]

    cmap = plt.get_cmap(
        "tab20"
    )  # tab20 has 20 distinct colors; better for categorical data
    colors = [
        cmap(i % 20) for i in range(num_particles)
    ]  # wrap around if >20 particles

    # Create a mapping: particle index -> color
    particle_colors = {i: colors[i] for i in range(x_particle_coordinates.shape[0])}

    fig, ax = plt.subplots(figsize=(12, 8))

    # Display the grid

    MAG_CPU = MAG.cpu()
    U_CPU = U.cpu()
    V_CPU = V.cpu()

    y_particle_coordinates_cpu = y_particle_coordinates.cpu()
    x_particle_coordinates_cpu = x_particle_coordinates.cpu()

    cax = ax.imshow(
        MAG_CPU[TIMESTEP, :, :],
        origin="lower",  # (0, 0) at the bottom left
        cmap="viridis",  # colormap
        extent=[
            0,
            MAG_CPU[TIMESTEP, :, :].shape[1] * GRID_SPACING_KM,
            0,
            MAG_CPU[TIMESTEP, :, :].shape[0] * GRID_SPACING_KM,
        ],  # [x_min, x_max, y_min, y_max]
        aspect="equal",
    )  # square pixel aspect ratio

    # Add a colorbar
    fig.colorbar(cax, ax=ax, label="Magnitude of Flow (Km/H)")

    # Label axes
    ax.set_xlabel("X coordinate (Km)")
    ax.set_ylabel("Y coordinate (Km)")
    ax.set_title(
        f"Simulation of particle movement superimposed onto \nthe magnitude of flow in the Phillipine Archipellago\n at Hour {(TIMESTEP) * 3} to {(TIMESTEP+1) * 3}"
    )

    # Plot the particle

    for particle in range(num_particles):
        # Convert Coordinate positions from index to Km
        x = x_particle_coordinates_cpu[particle][TIMESTEP]
        y = y_particle_coordinates_cpu[particle][TIMESTEP]

        # Plot Position
        ax.scatter(
            x,
            y,
            color=particle_colors[particle],
            s=20,
            label=f"Particle {particle + 1}",
        )

        # Plot Current Vector
        ax.arrow(
            x,
            y,
            ARROW_SCALE
            * U_CPU[
                TIMESTEP,
                (y_particle_coordinates_cpu[particle][TIMESTEP] / 3).int(),
                (x_particle_coordinates_cpu[particle][TIMESTEP] / 3).int(),
            ],
            ARROW_SCALE
            * V_CPU[
                TIMESTEP,
                (y_particle_coordinates_cpu[particle][TIMESTEP] / 3).int(),
                (x_particle_coordinates_cpu[particle][TIMESTEP] / 3).int(),
            ],
            head_width=20,
            head_length=20,
            fc=colors[particle],
            ec=colors[particle],
        )

        # Plot particle trace
        # Plot Position
        if TIMESTEP > 0:
            for past_index in range(0, TIMESTEP):
                x_past = x_particle_coordinates_cpu[particle][past_index]
                y_past = y_particle_coordinates_cpu[particle][past_index]

                ax.scatter(
                    x_past, y_past, color=particle_colors[particle], alpha=0.1, s=1
                )

    fig.text(
        0.1,
        0.02,
        "* Trajectory vectors are scaled by 2 orders of magnitude for visibility",
        fontsize=10,
        ha="left",
    )

    # Add legend
    ax.legend()

    plt.show()


def plot_particle_debris_movement(
    sampled_x, sampled_y, MAG, U, V, TIMESTEP=0, ARROW_SCALE=1
):
    """This function will simulate particle debris movement.

    Parameters
    ----------
    sampled_x_cpu : torch.Tensor
        This is the sample_x.
        The units are expected
        to already be in
        KILOMETERS.
    sampled_y_cpu : torch.Tensor
        This is the sample_y.
        The units are expected
        to already be in
        KILOMETERS.
    TIMESTEP : int, optional
        This is the timestep to
        plot the function,
        defaults to 0.
    ARROW_SCALE : int, optional
        This is the factor by which to scale the particle debris
        future trajectory for visibility,
        defaults to 1.
    MAG : torch.float64
        This is the tensor of combined magnitudes
        of flow vectors in the Philippine Archipelago.
    U : torch.float64
        This is the tensor of the horizontal component
        of flow vectors in the Philippine Archipelago.
    V : torch.float64
        This is the tensor of the vertical component
        of flow vectors in the Philippine Archipelago.
    """

    MAG_CPU = MAG.cpu()
    sampled_x_cpu = sampled_x.cpu()
    sampled_y_cpu = sampled_y.cpu()
    V_CPU = V.cpu()
    U_CPU = U.cpu()

    num_particles = sampled_x_cpu.shape[0]

    cmap = plt.get_cmap(
        "tab20"
    )  # tab20 has 20 distinct colors; better for categorical data
    colors = [
        cmap(i % 20) for i in range(num_particles)
    ]  # wrap around if >20 particles

    # Create a mapping: particle index -> color
    particle_colors = {i: colors[i] for i in range(sampled_x_cpu.shape[0])}

    fig, ax = plt.subplots(figsize=(12, 8))

    # Display the grid: MAG_CPU is a global variable of the calculated MAG_CPU loaded at folder initialization
    cax = ax.imshow(
        MAG_CPU[TIMESTEP, :, :],
        origin="lower",  # (0, 0) at the bottom left
        cmap="viridis",  # colormap
        extent=[
            0,
            MAG_CPU[TIMESTEP, :, :].shape[1] * GRID_SPACING_KM,
            0,
            MAG_CPU[TIMESTEP, :, :].shape[0] * GRID_SPACING_KM,
        ],  # [x_min, x_max, y_min, y_max]
        aspect="equal",
    )  # square pixel aspect ratio

    # Add a colorbar
    fig.colorbar(cax, ax=ax, label="Magnitude of Flow (Km/H)")

    # Label axes
    ax.set_xlabel("X coordinate (Km)")
    ax.set_ylabel("Y coordinate (Km)")
    ax.set_title(
        f"Debris Potential Locations at Hour {(TIMESTEP) * 3} to {(TIMESTEP+1) * 3}"
    )

    # Plot the particle
    for particle in range(10):
        x = sampled_x_cpu[particle][TIMESTEP]
        y = sampled_y_cpu[particle][TIMESTEP]

        # Plot Position
        ax.scatter(
            x,
            y,
            color=particle_colors[particle],
            s=20,
            label=f"Particle {particle + 1}: ({x:0f}, {y:0f})",
        )

        # Plot Current Vector
        ax.arrow(
            x,
            y,
            ARROW_SCALE
            * U_CPU[
                TIMESTEP,
                (sampled_y_cpu[particle][TIMESTEP] / 3).int(),
                (sampled_x_cpu[particle][TIMESTEP] / 3).int(),
            ],
            ARROW_SCALE
            * V_CPU[
                TIMESTEP,
                (sampled_y_cpu[particle][TIMESTEP] / 3).int(),
                (sampled_x_cpu[particle][TIMESTEP] / 3).int(),
            ],
            head_width=2,
            head_length=2,
            fc=particle_colors[particle],
            ec=particle_colors[particle],
        )

        # Plot particle trace

        # Plot Position
        if TIMESTEP > 0:
            for past_index in range(0, TIMESTEP):
                x_past = sampled_x_cpu[particle][past_index]
                y_past = sampled_y_cpu[particle][past_index]

                ax.scatter(
                    x_past, y_past, color=particle_colors[particle], alpha=0.8, s=1
                )

    fig.text(
        0.1,
        0.02,
        "* Trajectory vectors are scaled by 1 order of magnitude for visibility",
        fontsize=10,
        ha="left",
    )

    # Add legend
    ax.legend()

    # Set the limits to zoom into a specific region
    ax.set_xlim(200, 400)  # Zoom into x-axis between 2 and 5
    ax.set_ylim(1000, 1200)  # Zoom into y-axis between -0.5 and 0.5

    plt.show()


# Define your kernel function for vectors
def kernel_multi_dimension(X, Y, sigma, ell):
    if len(X.shape) <= 1:
        X = np.atleast_2d(X).T  # Make X a column vector
        Y = np.atleast_2d(Y).T  # Make Y a column vector
    sqdist = np.sum(X**2, 1).reshape(-1, 1) + np.sum(Y**2, 1) - 2 * X @ Y.T
    return sigma**2 * np.exp(-0.5 * sqdist / ell**2)


def create_partitions(U, y_coordinate, x_coordinate):
    kf = KFold(n_splits=10, shuffle=False)
    U_partitions = []

    # Iterate over the splits
    for fold, (train_idx, test_idx) in enumerate(kf.split(torch.arange(U.size(0)))):
        U_train_data = U[train_idx, y_coordinate, x_coordinate]
        U_test_data = U[test_idx, y_coordinate, x_coordinate]
        U_partitions.append({"fold": fold, "train": U_train_data, "test": U_test_data})
    return U_partitions


def rbf_kernel(
    X1: torch.Tensor, X2: torch.Tensor, sigma: float, ell: float
) -> torch.Tensor:
    """
    Compute the RBF (squared exponential) kernel matrix.

    K(x, x') = sigma^2 * exp(-||x - x'||^2 / (2 * ell^2))
    """
    sq_dist = torch.cdist(X1, X2, p=2) ** 2
    return sigma**2 * torch.exp(-0.5 * sq_dist / ell**2)


def compute_conditional(U_train, U_test, sigma, lengthscale, tau, verbose_tau):
    K_train = rbf_kernel(U_train, U_train, sigma, lengthscale)
    K_test = rbf_kernel(U_test, U_test, sigma, lengthscale)
    K_cross = rbf_kernel(U_test, U_train, sigma, lengthscale)

    # Adding noise for stability
    noise = tau * torch.eye(K_train.shape[0], device=device)
    K_train += noise

    # If this value is larger than ~1e10, you might want to increase tau to stabilize the inversion.
    if verbose_tau:
        K_condition = torch.linalg.cond(K_train)
        if K_condition > 1e10:
            print("Condition number of K_train:", K_condition.item())

    L = torch.linalg.cholesky(K_train)
    mu_train = torch.mean(U_train)
    mu_test = torch.mean(U_test)
    y_centered = U_train - mu_train

    alpha = torch.cholesky_solve(y_centered, L)
    cond_mean = mu_test + (K_cross @ alpha).squeeze(1)

    v = torch.cholesky_solve(K_cross.T, L)
    cond_cov = K_test - K_cross @ v

    return cond_mean, cond_cov


def compute_log_likelihood(x_test, mu_pred, cov_pred, tau):
    n = x_test.shape[0]
    cov_pred += tau * torch.eye(n, device=cov_pred.device)
    diff = (x_test - mu_pred).unsqueeze(-1)
    try:
        chol = torch.linalg.cholesky(cov_pred)
        log_det = 2 * torch.sum(torch.log(torch.diag(chol)))
        solve = torch.cholesky_solve(diff, chol)
        ll = -0.5 * (
            diff.T @ solve + log_det + n * torch.log(torch.tensor(2 * torch.pi))
        )
        return ll.item()
    except RuntimeError:
        return float("-inf")


def compute_ll_for_var_lengthscale_of_U_and_V_component(
    U, V, sigma_l, ell, tau, x_coordinate, y_coordinate, verbose_tau=False
):
    kf = KFold(n_splits=10, shuffle=False)
    indices = list(kf.split(torch.arange(U.size(0))))
    param_grid = list(product(sigma_l, ell))

    U_ll_dict = {}
    V_ll_dict = {}
    counts = {}

    y_coordinate_int = y_coordinate.int()
    x_coordinate_int = x_coordinate.int()

    for train_idx, test_idx in indices:
        train_idx = torch.tensor(train_idx, dtype=torch.int, device=device)
        test_idx = torch.tensor(test_idx, dtype=torch.int, device=device)

        # Extract and reshape into 2D column vectors: [n_samples, 1]
        U_train = U[train_idx, y_coordinate_int, x_coordinate_int].unsqueeze(1)
        U_test = U[test_idx, y_coordinate_int, x_coordinate_int].unsqueeze(1)
        V_train = V[train_idx, y_coordinate_int, x_coordinate_int].unsqueeze(1)
        V_test = V[test_idx, y_coordinate_int, x_coordinate_int].unsqueeze(1)

        for sigma, lengthscale in param_grid:
            mu_u, cov_u = compute_conditional(
                U_train, U_test, sigma, lengthscale, tau, verbose_tau
            )
            ll_u = compute_log_likelihood(U_test.squeeze(1), mu_u, cov_u, tau)

            mu_v, cov_v = compute_conditional(
                V_train, V_test, sigma, lengthscale, tau, verbose_tau
            )
            ll_v = compute_log_likelihood(V_test.squeeze(1), mu_v, cov_v, tau)

            key = (sigma, lengthscale)
            U_ll_dict[key] = U_ll_dict.get(key, 0.0) + ll_u
            V_ll_dict[key] = V_ll_dict.get(key, 0.0) + ll_v
            counts[key] = counts.get(key, 0) + 1

    for key in U_ll_dict:
        U_ll_dict[key] /= counts[key]
        V_ll_dict[key] /= counts[key]

    u_best_key = max(U_ll_dict.items(), key=lambda item: item[1])[0]
    v_best_key = max(V_ll_dict.items(), key=lambda item: item[1])[0]

    return (
        u_best_key[0],
        u_best_key[1],
        U_ll_dict[u_best_key],
        v_best_key[0],
        v_best_key[1],
        V_ll_dict[v_best_key],
    )


def logpdf_normal(
    x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    return (
        -0.5 * torch.log(torch.tensor(2 * torch.pi))
        - torch.log(sigma)
        - ((x - mu) ** 2) / (2 * sigma**2)
    )


# Example Use:
# x: y_test_U, mu: y_pred_U, sigma: y_std_U
# log_probs = logpdf_normal(y_test_U, y_pred_U, y_std_U)
# test_ll = log_probs.sum()


def simulate_expanded_particle_movement(
    n_particles,
    U,
    V,
    MAG,
    x_particle_coordinates_init=None,
    y_particle_coordinates_init=None,
    sigma=None,
    lengthscale=None,
):
    # Expanded time dimension (3x expansion)
    GRID_SPACING_KM = 3  # Define your grid spacing in kilometers
    expanded_timesteps = 300
    height, width = MAG.shape[1], MAG.shape[2]
    height_km = height * GRID_SPACING_KM
    width_km = width * GRID_SPACING_KM

    x_particle_coordinates = torch.empty(
        (n_particles, expanded_timesteps), dtype=torch.float64, device=device
    )
    y_particle_coordinates = torch.empty(
        (n_particles, expanded_timesteps), dtype=torch.float64, device=device
    )

    if x_particle_coordinates_init is None:
        x_particle_coordinates[:, 0] = torch.randint(
            0, width_km, (n_particles,), device=device
        )
    else:
        x_particle_coordinates[:, 0] = x_particle_coordinates_init.squeeze()

    if y_particle_coordinates_init is None:
        y_particle_coordinates[:, 0] = torch.randint(
            0, height_km, (n_particles,), device=device
        )
    else:
        y_particle_coordinates[:, 0] = y_particle_coordinates_init.squeeze()

    U_expanded = torch.full(
        (expanded_timesteps, height, width), torch.nan, dtype=torch.float64
    )
    V_expanded = torch.full(
        (expanded_timesteps, height, width), torch.nan, dtype=torch.float64
    )

    U_expanded[::3, :, :] = U
    V_expanded[::3, :, :] = V

    for t in tqdm(range(1, 300), desc="Simulating Particle Movement...", ascii="░▒▓█"):
        for particle in range(x_particle_coordinates.shape[0]):
            x_prev = x_particle_coordinates[particle, t - 1]
            y_prev = y_particle_coordinates[particle, t - 1]

            # Bounds check (in kilometers)
            if x_prev < 0 or y_prev < 0 or x_prev >= width_km or y_prev >= height_km:
                x_particle_coordinates[particle, t] = x_prev
                y_particle_coordinates[particle, t] = y_prev
                continue

            for hour in range(24):
                if (
                    x_prev < 0
                    or y_prev < 0
                    or x_prev >= width_km
                    or y_prev >= height_km
                ):
                    break  # Exit hour loop for this particle if it's now out of bounds
                x_idx = int(x_prev.item() / GRID_SPACING_KM)
                y_idx = int(y_prev.item() / GRID_SPACING_KM)
                if torch.isnan(U_expanded[t - 1, y_idx, x_idx]):
                    U_future, V_future = (
                        find_future_magnitude_given_coordinates_in_kilometers(
                            x_prev, y_prev, U, V, sigma=sigma, lengthscale=lengthscale
                        )
                    )
                    U_expanded[:, y_idx, x_idx] = U_future.flatten()
                    V_expanded[:, y_idx, x_idx] = V_future.flatten()

                x_prev += U_expanded[t - 1, y_idx, x_idx]
                y_prev += V_expanded[t - 1, y_idx, x_idx]
            x_particle_coordinates[particle, t] = x_prev
            y_particle_coordinates[particle, t] = y_prev

    return x_particle_coordinates, y_particle_coordinates, U_expanded, V_expanded


def simulate_expanded_particle_debris_scattering(U, V, MAG, n_particles=1):
    # Expanded time dimension (3x expansion)
    GRID_SPACING_KM = 3  # Define your grid spacing in kilometers
    expanded_timesteps = 300
    height, width = MAG.shape[1], MAG.shape[2]
    height_km = height * GRID_SPACING_KM
    width_km = width * GRID_SPACING_KM

    # Identify random means for the defined number of particles
    # with a uniform distribution of variances of debris

    # Define Mask & Valid Water Locations
    mask = pd.read_csv("./OceanFlow/mask.csv").to_numpy()
    # mask is upside-down
    mask = np.flipud(mask)
    water_locations_km = np.argwhere(mask == 1) * GRID_SPACING_KM
    print(mask.shape)
    random_water_loc_km_mean_idx = np.random.randint(
        0, water_locations_km.shape[0], (n_particles,)
    )

    # Allocate arrays
    x_particle_coordinates = np.zeros(
        (n_particles, expanded_timesteps), dtype=np.float64
    )
    y_particle_coordinates = np.zeros(
        (n_particles, expanded_timesteps), dtype=np.float64
    )
    x_particle_coordinates[:, 0] = water_locations_km[random_water_loc_km_mean_idx][
        :, 1
    ]
    y_particle_coordinates[:, 0] = water_locations_km[random_water_loc_km_mean_idx][
        :, 0
    ]

    # Convert to torch tensors on CUDA
    x_particle_coordinates = torch.from_numpy(x_particle_coordinates).to("cuda")
    y_particle_coordinates = torch.from_numpy(y_particle_coordinates).to("cuda")

    # Preallocate for the expansion of the U & V matrices
    U_expanded = torch.full(
        (expanded_timesteps, height, width), torch.nan, dtype=torch.float64
    )
    V_expanded = torch.full(
        (expanded_timesteps, height, width), torch.nan, dtype=torch.float64
    )

    U_expanded[::3, :, :] = U
    V_expanded[::3, :, :] = V

    for t in tqdm(range(1, 300), desc="Simulating Particle Movement...", ascii="░▒▓█"):
        for particle in range(x_particle_coordinates.shape[0]):
            x_prev = x_particle_coordinates[particle, t - 1]
            y_prev = y_particle_coordinates[particle, t - 1]

            # Bounds check (in kilometers)
            if x_prev < 0 or y_prev < 0 or x_prev >= width_km or y_prev >= height_km:
                x_particle_coordinates[particle, t] = x_prev
                y_particle_coordinates[particle, t] = y_prev
                continue

            for hour in range(24):
                if (
                    x_prev < 0
                    or y_prev < 0
                    or x_prev >= width_km
                    or y_prev >= height_km
                ):
                    break  # Exit hour loop for this particle if it's now out of bounds
                x_idx = int(x_prev.item() / GRID_SPACING_KM)
                y_idx = int(y_prev.item() / GRID_SPACING_KM)
                if torch.isnan(U_expanded[t - 1, y_idx, x_idx]):
                    U_future, V_future = (
                        find_future_magnitude_given_coordinates_in_kilometers(
                            x_prev, y_prev, U, V
                        )
                    )
                    U_expanded[:, y_idx, x_idx] = U_future.flatten()
                    V_expanded[:, y_idx, x_idx] = V_future.flatten()

                x_prev += U_expanded[t - 1, y_idx, x_idx]
                y_prev += V_expanded[t - 1, y_idx, x_idx]
            x_particle_coordinates[particle, t] = x_prev
            y_particle_coordinates[particle, t] = y_prev

    return x_particle_coordinates, y_particle_coordinates, U_expanded, V_expanded


def plot_particle_coordinates_traces_and_trajectories_for_timestep_days(
    x_particle_coordinates,
    y_particle_coordinates,
    MAG,
    U_expanded,
    V_expanded,
    DAY=1,
    ARROW_SCALE=72,
    search_locations=None,
    show_legend=False,
    use_imputed_arrows=False,
    save_figure_filename=None,
):
    """This function plots the particle positions, trace of past positions, and trajectory of future positions.

    Parameters
    ----------
    DAY : int
        This is the timestep (time index) for which to plot the data.
        Each index is zero-based.
        This parameter expectes the exact day 1-based.
        There are 300 days computed from 100 timesteps with 200 predictions.
        Each timestep is computed for 24 hours for the magnitude velocity.
    ARROW_SCALE : int
        This is the factor to scale the trajectory arrows by for visibility purposes.
    x_particle_coordinates : torch.tensor (n_particles, n_timesteps)
        These are the x coordinates of the particle.
    y_particle_coordinates : torch.tensor (n_particles, n_timesteps)
        These are the y coordinates of each particle.
    MAG: torch.float64
        This is the magnitude dataframe of the combined flows of the
        U and V matrices.
    U_expanded: torch.float64
        This is the U component (horizontal) flow.
    V_expanded: torch.float64
        This is the V component (vertical) flow expanded for a calculated
        timestep at each y, x coordindinate that have also been expanded
        to 300 for 1 day at each index. Every time index is calculated for
        each particle's y, x coordinates. Used for trajectory calculations
        and arrow scaling.
    search_locations: numpy.array, default None
        This is the list of locations to search in the form
        land search location: (x, y),
        ocean search location: (x, y)
        for indexes (0, 1) and (2, 3) respectively.
    show_legend: bool, default False
        This enables or disables plotting the legend.
    use_imputed_arrows: bool, default False
        This will use the extended Magnitude matrix to compute the pseudo-trajectories
        for missing coordinate locations that have been imputed. If the coordinates
        have been imputed, then there will not always be a magnitude for a particular
        coordinate pair. In such a case, the coordinate trajectory will need to be
        imputed.
    save_figure_filename: str, default None
        If a filename is passed to this function,
        the output plot will be saved to a directory
        with this filename.
    """

    # zero-based indexing
    DAY = DAY - 1

    GRID_SPACING_KM = 3
    U_expanded_CPU = U_expanded.cpu()
    V_expanded_CPU = V_expanded.cpu()

    # Magnitude plot is repeated for every value for 3 more time indices
    MAG_CPU = MAG.cpu()
    MAG_CPU_EXTENDED = torch.empty((300, 504, 555), device="cpu", dtype=MAG_CPU.dtype)
    for index in range(3):
        MAG_CPU_EXTENDED[index::3, :, :] = MAG_CPU

    num_particles = x_particle_coordinates.shape[0]

    cmap = plt.get_cmap(
        "tab20"
    )  # tab20 has 20 distinct colors; better for categorical data
    colors = [
        cmap(i % 20) for i in range(num_particles)
    ]  # wrap around if >20 particles

    # Create a mapping: particle index -> color
    particle_colors = {i: colors[i] for i in range(x_particle_coordinates.shape[0])}

    fig, ax = plt.subplots(figsize=(12, 8))

    # Display the grid
    y_particle_coordinates_cpu = y_particle_coordinates.cpu()
    x_particle_coordinates_cpu = x_particle_coordinates.cpu()

    cax = ax.imshow(
        MAG_CPU_EXTENDED[DAY, :, :],
        origin="lower",  # (0, 0) at the bottom left
        cmap="viridis",  # colormap
        extent=[
            0,
            MAG_CPU_EXTENDED[DAY, :, :].shape[1] * GRID_SPACING_KM,
            0,
            MAG_CPU_EXTENDED[DAY, :, :].shape[0] * GRID_SPACING_KM,
        ],  # [x_min, x_max, y_min, y_max]
        aspect="equal",
    )  # square pixel aspect ratio

    # Add a colorbar
    fig.colorbar(cax, ax=ax, label="Magnitude of Flow (Km/H)")

    # Label axes
    ax.set_xlabel("X coordinate (Km)")
    ax.set_ylabel("Y coordinate (Km)")
    ax.set_title(
        f"Simulation of particle movement superimposed onto \nthe magnitude of flow in the Phillipine Archipellago\n at Day {DAY+1}."
    )

    # Optimized plotting

    # Current positions (at DAY)
    x_current = x_particle_coordinates_cpu[:, DAY]
    y_current = y_particle_coordinates_cpu[:, DAY]

    # Plot all particles at once
    ax.scatter(
        x_current,
        y_current,
        c=[particle_colors[i] for i in range(num_particles)],
        s=20,
    )

    # Convert coordinates to indices
    x_idx = (x_current / GRID_SPACING_KM).long()
    y_idx = (y_current / GRID_SPACING_KM).long()

    # Vector directions
    u = ARROW_SCALE * U_expanded_CPU[DAY, y_idx, x_idx]
    v = ARROW_SCALE * V_expanded_CPU[DAY, y_idx, x_idx]

    # Plot arrows in batch
    ax.quiver(
        x_current,
        y_current,
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=1,
        color=[particle_colors[i] for i in range(num_particles)],
        width=0.003,
        headwidth=3,
        headlength=4,
    )

    if DAY > 1:
        lines = []
        trace_colors = []

        for particle in range(num_particles):
            x_trace = x_particle_coordinates_cpu[particle, :DAY]
            y_trace = y_particle_coordinates_cpu[particle, :DAY]
            points = np.array([x_trace, y_trace]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lines.extend(segments)
            trace_colors.extend([particle_colors[particle]] * (DAY - 1))

        trace_collection = LineCollection(
            lines, colors=trace_colors, linewidths=0.5, alpha=0.8
        )
        ax.add_collection(trace_collection)

    if search_locations is not None:
        ax.scatter(
            search_locations[0],
            search_locations[1],
            color="y",
            s=20,
            marker="x",
            alpha=0.8,
            label=f"Land Search Location",
        )
        ax.scatter(
            search_locations[2],
            search_locations[3],
            color="y",
            s=20,
            marker="s",
            alpha=0.8,
            label=f"Ocean Search Location",
        )

    fig.text(
        0.1,
        0.02,
        "* Trajectory vectors are projected by 72 hours for visibility",
        fontsize=10,
        ha="left",
    )

    # Add legend
    if show_legend:
        ax.legend()

    # Optional: Save the output plot to plots/save_figure_filename
    if save_figure_filename is not None:
        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, save_figure_filename)
        fig.savefig(filepath)

    plt.show()
    plt.close(fig)


def plot_particle_coordinates_traces_and_trajectories_for_timestep_days_monitoring_stations(
    x_particle_coordinates,
    y_particle_coordinates,
    MAG,
    U_expanded,
    V_expanded,
    DAY=1,
    ARROW_SCALE=72,
    stations=None,
    save_figure_filename=None,
):
    """This function plots the particle positions, trace of past positions, and trajectory of future positions.

    Parameters
    ----------
    DAY : int
        This is the timestep (time index) for which to plot the data.
        Each index is zero-based.
        This parameter expectes the exact day 1-based.
        There are 300 days computed from 100 timesteps with 200 predictions.
        Each timestep is computed for 24 hours for the magnitude velocity.
    ARROW_SCALE : int
        This is the factor to scale the trajectory arrows by for visibility purposes.
    x_particle_coordinates : torch.tensor (n_particles, n_timesteps)
        These are the x coordinates of the particle.
    y_particle_coordinates : torch.tensor (n_particles, n_timesteps)
        These are the y coordinates of each particle.
    MAG: torch.float64
        This is the magnitude dataframe of the combined flows of the
        U and V matrices.
    U_expanded: torch.float64
        This is the U component (horizontal) flow.
    V_expanded: torch.float64
        This is the V component (vertical) flow expanded for a calculated
        timestep at each y, x coordindinate that have also been expanded
        to 300 for 1 day at each index. Every time index is calculated for
        each particle's y, x coordinates. Used for trajectory calculations
        and arrow scaling.
    save_figure_filename: str, default None
        If a filename is passed to this function,
        the output plot will be saved to a directory
        with this filename.
    """

    # zero-based indexing
    DAY = DAY - 1

    GRID_SPACING_KM = 3
    U_expanded_CPU = U_expanded.cpu()
    V_expanded_CPU = V_expanded.cpu()

    # Magnitude plot is repeated for every value for 3 more time indices
    MAG_CPU = MAG.cpu()
    MAG_CPU_EXTENDED = torch.empty((300, 504, 555), device="cpu", dtype=MAG_CPU.dtype)
    for index in range(3):
        MAG_CPU_EXTENDED[index::3, :, :] = MAG_CPU

    num_particles = x_particle_coordinates.shape[0]

    cmap = plt.get_cmap(
        "tab20"
    )  # tab20 has 20 distinct colors; better for categorical data
    colors = [
        cmap(i % 20) for i in range(num_particles)
    ]  # wrap around if >20 particles

    # Create a mapping: particle index -> color
    particle_colors = {i: colors[i] for i in range(x_particle_coordinates.shape[0])}

    fig, ax = plt.subplots(figsize=(12, 8))

    # Display the grid
    y_particle_coordinates_cpu = y_particle_coordinates.cpu()
    x_particle_coordinates_cpu = x_particle_coordinates.cpu()

    cax = ax.imshow(
        MAG_CPU_EXTENDED[DAY, :, :],
        origin="lower",  # (0, 0) at the bottom left
        cmap="viridis",  # colormap
        extent=[
            0,
            MAG_CPU_EXTENDED[DAY, :, :].shape[1] * GRID_SPACING_KM,
            0,
            MAG_CPU_EXTENDED[DAY, :, :].shape[0] * GRID_SPACING_KM,
        ],  # [x_min, x_max, y_min, y_max]
        aspect="equal",
    )  # square pixel aspect ratio

    # Add a colorbar
    fig.colorbar(cax, ax=ax, label="Magnitude of Flow (Km/H)")

    # Label axes
    ax.set_xlabel("X coordinate (Km)")
    ax.set_ylabel("Y coordinate (Km)")
    ax.set_title(
        f"Simulation of particle movement superimposed onto \nthe magnitude of flow in the Phillipine Archipellago\n at Day {DAY+1}."
    )

    # Optimized plotting

    # Current positions (at DAY)
    x_current = x_particle_coordinates_cpu[:, DAY]
    y_current = y_particle_coordinates_cpu[:, DAY]

    # Plot all particles at once
    ax.scatter(
        x_current,
        y_current,
        c=[particle_colors[i] for i in range(num_particles)],
        s=20,
    )

    # Convert coordinates to indices
    x_idx = (x_current / GRID_SPACING_KM).long()
    y_idx = (y_current / GRID_SPACING_KM).long()

    # Vector directions
    u = ARROW_SCALE * U_expanded_CPU[DAY, y_idx, x_idx]
    v = ARROW_SCALE * V_expanded_CPU[DAY, y_idx, x_idx]

    # Plot arrows in batch
    ax.quiver(
        x_current,
        y_current,
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=1,
        color=[particle_colors[i] for i in range(num_particles)],
        width=0.003,
        headwidth=3,
        headlength=4,
    )

    if DAY > 1:
        lines = []
        trace_colors = []

        for particle in range(num_particles):
            x_trace = x_particle_coordinates_cpu[particle, :DAY]
            y_trace = y_particle_coordinates_cpu[particle, :DAY]
            points = np.array([x_trace, y_trace]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lines.extend(segments)
            trace_colors.extend([particle_colors[particle]] * (DAY - 1))

        trace_collection = LineCollection(
            lines, colors=trace_colors, linewidths=0.5, alpha=0.8
        )
        ax.add_collection(trace_collection)

    if stations is not None:
        station_labels = ["Station 1", "Station 2", "Station 3"]
        markers = ["x", "2", "^"]

        for i in range(3):
            x_coord = stations[2 * i]
            y_coord = stations[2 * i + 1]
            ax.scatter(
                x_coord,
                y_coord,
                color="y",
                s=30,
                marker=markers[i],
                alpha=0.8,
                label=station_labels[i],
            )

    fig.text(
        0.1,
        0.02,
        "* Trajectory vectors are projected by 72 hours for visibility",
        fontsize=10,
        ha="left",
    )

    # Add legend
    ax.legend()

    # Optional: Save the output plot to plots/save_figure_filename
    if save_figure_filename is not None:
        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, save_figure_filename)
        fig.savefig(filepath)

    plt.show()
    plt.close(fig)
