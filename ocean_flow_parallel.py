import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
import torch
from sklearn.model_selection import KFold
from itertools import product
import em
import common

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Global Variables
DATA_DIR = "OceanFlow/"
NUM_TIMESTEPS = 100  # T = 1 to 100
GRID_SPACING_KM = 3


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


def find_future_magnitude_given_coordinates_in_kilometers(
    x_point_km, y_point_km, U, V, sigma=None, lengthscale=None
):
    """This function will accept two points in units of
    kilometers and return the future expanded magnitude
    for all 300 days at that coordinate pair.

    Parameters
    ----------
    x_point_km : numpy.ndarray
        This is the x coordinate in kilometers to
          identify the future expanded magnitude flow vector.
    y_point_km : numpy.ndarray
        This is the y coordinate in kilometers to
          identify the future expanded magnitude flow vector.
    U : torch.tensor
        This is the horizontal component of flow.
    V : torch.tensor
        This is the vertical component of flow.
    sigma : torch.float64
        This is the generally-applicable hyperparameter selected
        to compute the conditional distribution.
    lengthscale : torch.float64
        This is the generally-applicable hyperparameter selected
        to compute the conditional distribution.
    """
    # Overarching process

    # Coodinates
    x_coordinate_scaled = x_point_km // 3
    y_coordinate_scaled = y_point_km // 3
    # print(f"x_coordinate, y_coordinate: {x_coordinate_scaled, y_coordinate_scaled}")

    if sigma is None and lengthscale is None:
        # Kernel Hyper-Parameters:
        variance = np.array([0.1, 1, 2, 3, 4, 5, 6])
        sigma_l = np.sqrt(variance)
        tau = 0.1
        # 7.2 hours, 36 hours, 72 hours, 144 hours, 288 hours
        # 7.2 hours, 1.5 days, 3 days, 6 days, 12 days
        ell = np.array([0.1, 0.5, 1, 2, 4])

        (
            u_best_sigma,
            u_best_lengthscale,
            u_best_log_likelihood,
            v_best_sigma,
            v_best_lengthscale,
            v_best_log_likelihood,
        ) = compute_ll_for_var_lengthscale_of_U_and_V_component(
            U, V, sigma_l, ell, tau, x_coordinate_scaled, y_coordinate_scaled
        )
    else:
        u_best_sigma = v_best_sigma = sigma
        u_best_lengthscale = v_best_lengthscale = lengthscale

    # Initializing the data
    U_time = U[:, y_coordinate_scaled.int(), x_coordinate_scaled.int()]
    V_time = V[:, y_coordinate_scaled.int(), x_coordinate_scaled.int()]

    U_three_days = torch.empty(300, device=device, dtype=torch.float64)
    U_three_days[[index for index in range(0, 300, 3)]] = U_time

    V_three_days = torch.empty(300, device=device)
    V_three_days[[index for index in range(0, 300, 3)]] = V_time

    U_X_test = torch.sort(
        torch.cat(
            [
                torch.arange(1, 300, 3, device=device, dtype=torch.float64),
                torch.arange(2, 300, 3, device=device, dtype=torch.float64),
            ]
        )
    )[0].reshape(-1, 1)
    U_X_train = torch.arange(0, 300, 3, device=device, dtype=torch.float64).reshape(
        -1, 1
    )
    U_y_train = U_time.reshape(-1, 1)

    V_X_test = torch.sort(
        torch.cat(
            [
                torch.arange(1, 300, 3, device=device, dtype=torch.float64),
                torch.arange(2, 300, 3, device=device, dtype=torch.float64),
            ]
        )
    )[0].reshape(-1, 1)
    V_X_train = torch.arange(0, 300, 3, device=device, dtype=torch.float64).reshape(
        -1, 1
    )
    V_y_train = V_time.reshape(-1, 1)

    U_test_mean, U_test_cov = predict_conditional_mean_variance_fixed_hyperparams(
        U_X_train,
        U_X_test,
        U_y_train,
        sigma=u_best_sigma,
        ell=u_best_lengthscale,
        tau=0.1,
    )
    V_test_mean, V_test_cov = predict_conditional_mean_variance_fixed_hyperparams(
        V_X_train,
        V_X_test,
        V_y_train,
        sigma=v_best_sigma,
        ell=v_best_lengthscale,
        tau=0.1,
    )

    U_expanded_pred = torch.empty((300, 1), device=device, dtype=torch.float64)
    U_expanded_pred[0::3] = U_y_train
    U_expanded_pred[1::3] = U_test_mean[::2]
    U_expanded_pred[2::3] = U_test_mean[1::2]

    V_expanded_pred = torch.empty((300, 1), device=device, dtype=torch.float64)
    V_expanded_pred[0::3] = V_y_train
    V_expanded_pred[1::3] = V_test_mean[::2]
    V_expanded_pred[2::3] = V_test_mean[1::2]

    return U_expanded_pred, V_expanded_pred


def simulate_expanded_particle_debris_scattering(
    U, V, MAG, n_particles=1, sigma=None, lengthscale=None, mask=None
):
    # Expanded time dimension (3x expansion)
    GRID_SPACING_KM = 3  # Define your grid spacing in kilometers
    expanded_timesteps = 300
    height, width = MAG.shape[1], MAG.shape[2]
    height_km = height * GRID_SPACING_KM
    width_km = width * GRID_SPACING_KM

    # Identify random means for the defined number of particles
    # with a uniform distribution of variances of debris
    if mask is None:
        # Define Mask & Valid Water Locations
        mask = pd.read_csv("./OceanFlow/mask.csv").to_numpy()
        # mask is upside-down
        mask = np.flipud(mask)
        # Assuming `mask` has shape (503, 555) and `coord_plane` has shape (504, 555)

        # Pad the mask at the bottom by 1 row (with zeros)
        mask = np.pad(mask, ((0, 1), (0, 0)), mode="constant", constant_values=0)
    water_locations_km = np.argwhere(mask == 1) * GRID_SPACING_KM
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


def simulate_expanded_imputed_particle_debris_scattering(
    U, V, MAG, n_particles_seed, n_particles_impute, sigma, lengthscale, mask=None
):
    """This function will accept a number of particles to seed, a larger number of
    particles to impute, and use the seed particles to identify the larger matrix of
    imputed particles by combining the two into a single matrix and using matrix
    completion methods such as gaussian mixture models with collaborative filtering
    and expectation maximization.

    Parameters
    ----------
    U : torch.float64
        This is the horizontal component of the flow velocities.
    V : torch.float64
        This is the vertical component of the flow velocities.
    MAG : torch.float64
        This is the combined and computed magnitude of the flow velocities.
    n_particles_seed : int
        This is the number of particles to brute-force compute.
    n_particles_impute : int
        This is the number of particles to impute after the brute-force
        computation.
    sigma : torch.float64
        This is the generally-applicable optimal hyperparameter
        applied to the RBF kernel in the brute-force compute.
    lengthscale : torch.float64
        This is generally-applicable optimal hyperparameter of
        the normalization factor of the RBF kernel used to regularize
        the predicted signal during brute-force prediction.
    mask : numpy.ndarray
        This is the mask for the coordinate grid.
        If read in with pandas and converted to numpy with
        to_numpy, the mask is the correct orientation.
        It is missing row of y-values and is padded to the
        correct shape on the bottom to become (504, 555).

    Returns
    -------
    torch.float64
        X_imputed_torch, y_imputed_torch are the completed flow velocities in km.
    """

    # Expanded time dimension (3x expansion)
    GRID_SPACING_KM = 3  # Define your grid spacing in kilometers
    expanded_timesteps = 300
    height, width = MAG.shape[1], MAG.shape[2]
    height_km = height * GRID_SPACING_KM
    width_km = width * GRID_SPACING_KM

    # Identify random means for the defined number of particles
    # with a uniform distribution of variances of debris
    # Define Mask & Valid Water Locations: 1 is water 0 is land
    if mask is None:
        mask = pd.read_csv("./OceanFlow/mask.csv").to_numpy()
        mask = np.flipud(mask)
        # Pad the mask at the bottom by 1 row (with zeros)
        mask = np.pad(mask, ((0, 1), (0, 0)), mode="constant", constant_values=0)
    water_locations_km = np.argwhere(mask == 1) * GRID_SPACING_KM
    random_water_loc_km_mean_idx = np.random.randint(
        0, water_locations_km.shape[0], (n_particles_impute,)
    )

    x_particle_coordinates, y_particle_coordinates, U_expanded, V_expanded = (
        simulate_expanded_particle_debris_scattering(
            U=U,
            V=V,
            MAG=MAG,
            n_particles=n_particles_seed,
            sigma=sigma,
            lengthscale=lengthscale,
            mask=mask,
        )
    )

    # Allocate arrays
    x_particle_coordinates_impute = np.zeros(
        (n_particles_impute, expanded_timesteps), dtype=np.float64
    )
    y_particle_coordinates_impute = np.zeros(
        (n_particles_impute, expanded_timesteps), dtype=np.float64
    )
    x_particle_coordinates_impute[:, 0] = water_locations_km[
        random_water_loc_km_mean_idx
    ][:, 1]
    y_particle_coordinates_impute[:, 0] = water_locations_km[
        random_water_loc_km_mean_idx
    ][:, 0]
    # Convert to torch tensors on CUDA
    x_particle_coordinates_impute = torch.from_numpy(x_particle_coordinates_impute).to(
        "cuda"
    )
    y_particle_coordinates_impute = torch.from_numpy(y_particle_coordinates_impute).to(
        "cuda"
    )

    x_particle_coordinates_incomplete = torch.cat(
        [x_particle_coordinates, x_particle_coordinates_impute], dim=0
    )
    y_particle_coordinates_incomplete = torch.cat(
        [y_particle_coordinates, y_particle_coordinates_impute], dim=0
    )
    y = y_particle_coordinates_incomplete.cpu().numpy()
    X = x_particle_coordinates_incomplete.cpu().numpy()

    # Impute X
    K = np.array(np.arange(1, 13))
    seeds = np.array([0, 1, 2, 3, 4])
    best_seeds = {}
    for k in tqdm(K, desc="Iterating K...", ascii="░▒▓█"):
        best_ll = np.inf
        best_seed = None
        for seed in tqdm(seeds, desc="Identifying best seed...", ascii="░▒▓█"):
            mixture, post = common.init(X=X, K=k, seed=seed)
            mixture, post, ll = tqdm(
                em.run(X=X, post=post, mixture=mixture),
                desc="Running EM Algorithm...",
                ascii="░▒▓█",
            )
            if ll < best_ll:
                best_ll = ll
                best_seed = seed
        # print(f"k:{k}")
        # print(f"best_seed:{best_seed}")
        # print(f"best_ll:{best_ll}")
        best_seeds[k] = dict({"seed": best_seed, "ll": best_ll})

        pd.DataFrame(best_seeds).T.reset_index()
        best_seeds_df = (
            pd.DataFrame(best_seeds)
            .T.reset_index()
            .rename(columns={"index": "K", "seed": "seed", "ll": "ll"})
        )
        best_K_seed = best_seeds_df.loc[
            best_seeds_df["ll"] == np.max(best_seeds_df["ll"])
        ]

        best_K = best_K_seed["K"].values[0]
        best_seed = int(best_K_seed["seed"].values[0])
        mixture, post = tqdm(
            common.init(X=X, K=best_K, seed=best_seed),
            desc="Initializing Mixture...",
            ascii="░▒▓█",
        )
        mixture, post, ll = tqdm(
            em.run(X=X, post=post, mixture=mixture),
            desc="Running EM Algorithm...",
            ascii="░▒▓█",
        )
        X_imputed = em.fill_matrix(X=X, mixture=mixture)

    # Impute y
    best_seeds = {}
    for k in tqdm(K, desc="Iterating K...", ascii="░▒▓█"):
        best_ll = np.inf
        best_seed = None
        for seed in tqdm(seeds, desc="Identifying best seed...", ascii="░▒▓█"):
            mixture, post = common.init(X=y, K=k, seed=seed)
            mixture, post, ll = tqdm(
                em.run(X=y, post=post, mixture=mixture),
                desc="Running EM Algorithm...",
                ascii="░▒▓█",
            )
            if ll < best_ll:
                best_ll = ll
                best_seed = seed
        # print(f"k:{k}")
        # print(f"best_seed:{best_seed}")
        # print(f"best_ll:{best_ll}")
        best_seeds[k] = dict({"seed": best_seed, "ll": best_ll})

        pd.DataFrame(best_seeds).T.reset_index()
        best_seeds_df = (
            pd.DataFrame(best_seeds)
            .T.reset_index()
            .rename(columns={"index": "K", "seed": "seed", "ll": "ll"})
        )
        best_K_seed = best_seeds_df.loc[
            best_seeds_df["ll"] == np.max(best_seeds_df["ll"])
        ]

        best_K = best_K_seed["K"].values[0]
        best_seed = int(best_K_seed["seed"].values[0])
        mixture, post = tqdm(
            common.init(X=y, K=best_K, seed=best_seed),
            desc="Initializing Mixture...",
            ascii="░▒▓█",
        )
        mixture, post, ll = tqdm(
            em.run(X=y, post=post, mixture=mixture),
            desc="Running EM Algorithm...",
            ascii="░▒▓█",
        )
        y_imputed = em.fill_matrix(X=y, mixture=mixture)
        return X_imputed, y_imputed, U_expanded, V_expanded


def remove_land_coordinates_and_beach_debris_X_imputed_y_imputed(
    X_imputed, y_imputed, U_expanded, V_expanded, mask=None
):
    """This function will remove coordinates that have been imputed to begin on land.
    It will allow the remaining coordinates to express "beaching" upon arriving at land
    by forward-filling their landcoordinate to the remainder of the time array.
    simulate_expanded_imputed_particle_debris_scattering must be called as the inputs
    are the return values from this function.

    Parameters
    ----------
    X_imputed : numpy.ndarray
        This is an array of shape n_particles_seed + n_particles_imputed.
        It is in kilometers.
    y_imputed : numpy.ndarray
        This is an array of shape n_particles_seed + n_particles_imputed.
        It is in kilometers.
    # U_expanded : torch.float64
    #     This is the matrix of the horizontal components of the
    #     imputed and extended flow velocities.
    # V_expanded : torch.float64
    #     This is the matrix of the vertical component of the
    #     imputed and extended flow velocities.
    mask : numpy.ndarray, default None
        If the land mask is not set, it will be properly created.

    Returns
    -------
    torch.float64
        X_imputed_torch, y_imputed_torch are tensors in km on the GPU.
    """
    # Identify random means for the defined number of particles
    # with a uniform distribution of variances of debris
    # Define Mask & Valid Water Locations: 1 is water 0 is land
    if mask is None:
        mask = pd.read_csv("./OceanFlow/mask.csv").to_numpy()
        mask = np.flipud(mask)
        # Pad the mask at the bottom by 1 row (with zeros)
        mask = np.pad(mask, ((0, 1), (0, 0)), mode="constant", constant_values=0)
    # Allowing imputed points to "beach" & removing land-starting locations
    # y_imputed_cpu_norm = np.float64(y_imputed) // GRID_SPACING_KM
    # X_imputed_cpu_norm = np.float64(X_imputed) // GRID_SPACING_KM

    # Example setup (you should load/compute your own)
    n_particles, n_time = X_imputed.shape[0], X_imputed.shape[1]
    GRID_SPACING_KM = 3

    # Convert to grid indices
    x = (X_imputed // GRID_SPACING_KM).astype(np.int64)
    y = (y_imputed // GRID_SPACING_KM).astype(np.int64)

    # Determine which particle is on land at each time
    on_land = mask[y, x] == 0  # shape (n_particles, n_time)

    # Get the first time a particle hits land (returns -1 if never)
    land_time = np.where(
        on_land.any(axis=1), on_land.argmax(axis=1), -1
    )  # shape (n_particles,)

    # Initialize output arrays (float to support np.nan if desired)
    x_masked = x.astype(float)
    y_masked = y.astype(float)

    # Now mask values after land_time
    for i in range(n_particles):
        lt = land_time[i]
        if lt != -1:
            # Option 1: Freeze at land point
            x_masked[i, lt:] = x[i, lt]
            y_masked[i, lt:] = y[i, lt]

            if lt == 0:
                # Option 2 (alternate): Mask with np.nan
                x_masked[i, lt:] = np.nan
                y_masked[i, lt:] = np.nan
    X_imputed_cpu_norm = pd.DataFrame(x_masked).dropna(axis=0).to_numpy()
    y_imputed_cpu_norm = pd.DataFrame(y_masked).dropna(axis=0).to_numpy()
    X_imputed_torch = torch.from_numpy(X_imputed_cpu_norm).to(device)
    y_imputed_torch = torch.from_numpy(y_imputed_cpu_norm).to(device)
    X_imputed_masked_torch_km = X_imputed_torch * GRID_SPACING_KM
    y_imputed_masked_torch_km = y_imputed_torch * GRID_SPACING_KM
    U_expanded_torch = U_expanded
    V_expanded_torch = V_expanded
    return (
        X_imputed_masked_torch_km,
        y_imputed_masked_torch_km,
        U_expanded_torch,
        V_expanded_torch,
    )
