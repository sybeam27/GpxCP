import numpy as np
import matplotlib.pyplot as plt

def check_sparsity(A):
    n_u, n_v = A.shape
    n = n_u + n_v
    E = A.sum()
    avg_deg = 2 * E / n
    density = E / (n_u * n_v)  # bipartite density
    
    return print("Average degree:", round(avg_deg, 4), "Density:", round(density, 4))

def evaluate_ci_bf(true_vals, lower, upper, clip=True): 
    if clip: 
        lower = np.clip(lower, 0, 1) 
        upper = np.clip(upper, 0, 1) 

    covered = (true_vals >= lower) & (true_vals <= upper) 
    coverage = covered.mean() 
    avg_width = np.mean(upper - lower) 
    return coverage, avg_width

def evaluate_ci_extended(true_vals, lower, upper, alpha=0.1, clip=True):
    """
    - Coverage
    - Avg Width
    - Normalized Interval Width (NIW)
    - Interval Score (Proper Scoring Rule)
    """
    if clip:
        lower = np.clip(lower, 0, 1)
        upper = np.clip(upper, 0, 1)

    covered = (true_vals >= lower) & (true_vals <= upper)
    coverage = covered.mean()
    widths = upper - lower
    avg_width = np.mean(widths)

    # Normalized Interval Width
    midpoints = (lower + upper) / 2
    niw = np.mean(widths / (np.abs(midpoints) + 1e-6))

    # Interval Score (Gneiting & Raftery, 2007)
    # S = width + (2/alpha)*(l - y) * 1(y<l) + (2/alpha)*(y - u) * 1(y>u)
    penalty_lower = (lower - true_vals) * (true_vals < lower)
    penalty_upper = (true_vals - upper) * (true_vals > upper)
    interval_score = np.mean(widths + (2/alpha)*(penalty_lower + penalty_upper))

    results = {
        "coverage": coverage,
        "avg_width": avg_width,
        "niw": niw,
        "interval_score": interval_score,
        "widths": widths, 
        "covered": covered
    }

    return results

def plot_width_distribution(widths, name="Dataset"):
    # Interval Width Distribution
    plt.figure(figsize=(6,4))
    plt.hist(widths, bins=30, color="skyblue", edgecolor="k", alpha=0.7)
    plt.xlabel("Interval Width")
    plt.ylabel("Frequency")
    plt.title(f"Interval Width Distribution ({name})")
    plt.show()

def plot_calibration_curve(true_vals, lower, upper, n_bins=10, name="Dataset"):
    """
    Nominal coverage vs Empirical coverage Curve
    # Calibration Curve
    """
    nominal_levels = np.linspace(0.05, 0.95, n_bins)
    empirical_coverages = []

    midpoints = (lower + upper) / 2
    widths = (upper - lower) / 2

    for alpha in nominal_levels:
        z = 1 - alpha
        # alpha-based narrower interval
        lower_alpha = midpoints - z * widths
        upper_alpha = midpoints + z * widths
        covered = (true_vals >= lower_alpha) & (true_vals <= upper_alpha)
        empirical_coverages.append(covered.mean())

    plt.figure(figsize=(5,5))
    plt.plot(nominal_levels, nominal_levels, "k--", label="Ideal")
    plt.plot(nominal_levels, empirical_coverages, "o-", label="Empirical")
    plt.xlabel("Nominal Coverage")
    plt.ylabel("Empirical Coverage")
    plt.title(f"Calibration Curve ({name})")
    plt.legend()
    plt.show()
