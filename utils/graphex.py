import numpy as np

def estimate_graphex_prob(A, alpha=0.1, p_star=0.05, p_noise=0.01):
    """
    Simple Graphex surrogate:
    - Kernel W: outer product of normalized degrees
    - Star S: row/col hubs (Top 10%)
    - Noise I: constant noise
    """
    n_u, n_v = A.shape
    deg_u = A.sum(axis=1) / n_v
    deg_v = A.sum(axis=0) / n_u
    
    W_kernel = np.outer(deg_u, deg_v) * alpha # Kernel component
    
    # Star component
    W_star = np.zeros_like(W_kernel)
    u_hubs = np.argsort(-deg_u)[:max(1, n_u//10)]
    v_hubs = np.argsort(-deg_v)[:max(1, n_v//10)]
    W_star[u_hubs, :] += p_star
    W_star[:, v_hubs] += p_star
    
    # Noise
    W_noise = np.full_like(W_kernel, p_noise)
    
    # Total probability
    W_total = np.clip(W_kernel + W_star + W_noise, 0, 1)
    return W_total

def estimate_robust_graphex_prob(A, alpha=0.1, p_star=0.05, p_noise=0.01, beta=0.2):
    """
    Noise-aware Graphex surrogate:
    W_total = (1-beta) * homophily + beta * heterophily + noise
    """
    n_u, n_v = A.shape
    deg_u = A.sum(axis=1) / n_v
    deg_v = A.sum(axis=0) / n_u

    # Homophily (outer product)
    W_homo = np.outer(deg_u, deg_v) * alpha

    # Heterophily: degree-mismatched edges (high-degree ↔ low-degree)
    W_hetero = np.outer(deg_u, 1 - deg_v) * alpha

    # Star + noise
    W_star = np.zeros_like(W_homo)
    u_hubs = np.argsort(-deg_u)[:max(1, n_u//10)]
    v_hubs = np.argsort(-deg_v)[:max(1, n_v//10)]
    W_star[u_hubs, :] += p_star
    W_star[:, v_hubs] += p_star

    W_noise = np.full_like(W_homo, p_noise)

    # Mixture
    W_total = (1-beta)*W_homo + beta*W_hetero + W_star + W_noise
    return np.clip(W_total, 0, 1)

def generate_bipartite_graphex(T=10, alpha=0.5, p_star=0.05, p_noise=0.01, seed=42):
    """
    Simulator for a finite-truncated bipartite network based on the Graphex process.
    
    Components:
    - W: kernel component
    - S: star component
    - I: independent random noise edges
    
    Parameters
    ----------
    T : float, optional
        Truncation parameter for the Poisson process (default=10).
    alpha : float, optional
        Scaling factor for the kernel component (default=0.5).
    p_star : float, optional
        Probability weight for the star component (default=0.05).
    p_noise : float, optional
        Independent noise edge probability (default=0.01).
    seed : int, optional
        Random seed for reproducibility (default=42).
    
    Returns
    -------
    A : np.ndarray
        Adjacency matrix (n_u x n_v).
    U : np.ndarray
        Node coordinates for the U partition (n_u x 1).
    V : np.ndarray
        Node coordinates for the V partition (n_v x 1).
    components : dict
        Dictionary containing intermediate matrices:
        - "W_kernel": kernel component matrix
        - "W_star": star component matrix
        - "A_noise": noise adjacency matrix
        - "W_total": total probability matrix before sampling
        - "A_main": adjacency matrix from kernel + star components
    """
    np.random.seed(seed)
    
    # 1. Poisson process for U, V node sampling
    n_u = np.random.poisson(T)
    n_v = np.random.poisson(T)
    U = np.random.rand(n_u, 1) * T
    V = np.random.rand(n_v, 1) * T
    
    # 2. Kernel component W(u,v)
    W_kernel = np.exp(-np.abs(U - V.T)) * alpha
    
    # 3. Star component (S)
    star_u_mask = np.random.binomial(1, 0.1, size=n_u)
    star_v_mask = np.random.binomial(1, 0.1, size=n_v)
    
    W_star = np.zeros_like(W_kernel)
    for i in range(n_u):
        if star_u_mask[i] == 1:
            W_star[i, :] += p_star
    for j in range(n_v):
        if star_v_mask[j] == 1:
            W_star[:, j] += p_star
    
    # 4. Independent noise component (I)
    A_noise = np.random.binomial(1, p_noise, size=(n_u, n_v))
    
    # 5. Total probability matrix
    W_total = W_kernel + W_star
    A_prob = np.clip(W_total, 0, 1)
    
    # 6. Adjacency matrix (Bernoulli sampling)
    A_main = np.random.binomial(1, A_prob)
    A = np.clip(A_main + A_noise, 0, 1)
    
    return A, U, V, {
        "W_kernel": W_kernel,
        "W_star": W_star,
        "A_noise": A_noise,
        "W_total": W_total,
        "A_main": A_main
    }
