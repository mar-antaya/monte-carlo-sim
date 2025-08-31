import os, time, math
import numpy as _np

try:
    import cupy as cp
    xp = cp
    on_gpu = True
except Exception:
    xp = _np
    on_gpu = False

def percentile(x, q):
    return cp.percentile(x, q) if on_gpu else _np.percentile(x, q)

def to_cpu(x):
    return cp.asnumpy(x) if on_gpu else x

def main(
    n_paths: int = 10_000_000,
    initial_value: float = 100_000.0,
    horizon_years: float = 1.0,
    seed: int = 42
):
    """
    Monte Carlo simulation for a portfolio modeled as GBM.
    Replace weights/mus/vols/corr with your portfolio.
    """

    # portfolio setup
    weights = xp.array([0.30, 0.25, 0.25, 0.20])   # must sum to 1
    mus_annual = xp.array([0.08, 0.05, 0.12, 0.03])
    vols_annual = xp.array([0.20, 0.15, 0.25, 0.10])

    # correlation matrix
    corr = xp.array([
        [1.00, 0.40, 0.30, 0.10],
        [0.40, 1.00, 0.35, 0.15],
        [0.30, 0.35, 1.00, 0.05],
        [0.10, 0.15, 0.05, 1.00],
    ])

    D = xp.diag(vols_annual)
    Sigma = D @ corr @ D

    mu_p = float(weights @ mus_annual)
    var_p = float(weights @ Sigma @ weights)
    sigma_p = math.sqrt(var_p)

    # random seed
    cp.random.seed(seed) if on_gpu else _np.random.seed(seed)

    t0 = time.time()
    drift_term = (mu_p - 0.5 * sigma_p**2) * horizon_years
    diffusion_std = sigma_p * math.sqrt(horizon_years)

    # simulate terminal values
    Z = xp.random.standard_normal(size=n_paths, dtype=xp.float32)
    terminal_vals = xp.asarray(initial_value, dtype=xp.float32) * xp.exp(
        drift_term + diffusion_std * Z
    )

    if on_gpu:
        cp.cuda.Stream.null.synchronize()

    sim_time = time.time() - t0

    # risk metrics
    losses = initial_value - terminal_vals
    mean_val = float(to_cpu(terminal_vals.mean()))
    std_val = float(to_cpu(terminal_vals.std()))

    var95 = float(to_cpu(percentile(losses, 95)))
    cvar95 = float(to_cpu(losses[losses >= var95].mean()))
    p_loss = float(to_cpu((terminal_vals < initial_value).mean()))

    p10 = float(to_cpu(percentile(terminal_vals, 10)))
    p50 = float(to_cpu(percentile(terminal_vals, 50)))
    p90 = float(to_cpu(percentile(terminal_vals, 90)))

    device = "GPU (CuPy/CUDA)" if on_gpu else "CPU (NumPy)"
    print("\n=== Monte Carlo Portfolio Risk (GBM, terminal-only) ===")
    print(f"Device:              {device}")
    print(f"Scenarios (n_paths): {n_paths:,}")
    print(f"Horizon (years):     {horizon_years}")
    print(f"Initial Value:       ${initial_value:,.2f}")
    print(f"mu_p (annual):       {mu_p:.4f}")
    print(f"sigma_p (annual):    {sigma_p:.4f}")
    print("\n--- Results ---")
    print(f"Mean terminal:       ${mean_val:,.2f}")
    print(f"Std terminal:        ${std_val:,.2f}")
    print(f"10th pct:            ${p10:,.2f}")
    print(f"Median:              ${p50:,.2f}")
    print(f"90th pct:            ${p90:,.2f}")
    print(f"VaR(95%):            ${var95:,.2f}   (positive = loss)")
    print(f"CVaR(95%):           ${cvar95:,.2f}  (avg loss in worst 5%)")
    print(f"P(Loss):             {p_loss*100:.2f}%")
    print(f"\nElapsed:             {sim_time:.2f} sec\n")

if __name__ == "__main__":
    n_paths = int(os.getenv("N_PATHS", "10000000"))
    horizon = float(os.getenv("HORIZON_YEARS", "1.0"))
    main(n_paths=n_paths, horizon_years=horizon)
