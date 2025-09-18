import numpy as np
import networkx as nx
import scipy.stats as st
import pymc as pm

from sklearn.utils import resample
from networkx.algorithms import bipartite
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import QuantileRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Bernoulli
from scipy.special import expit
import torch.nn.functional as F


# Conformal Prediction와 비교하기 위해서는 test set 전체에 공통적으로 적용할 수 있는 “규칙(구간 폭)”이 필요

# ------------ Resampling ------------ #
# bootstrap: 데이터를 여러 번 재표본화해서 통계량의 분포를 근사하는 방법, with replacement / Node 중심성 분포 불확실성 반영
def bootstrap_ci(A, cal_idx, n_boot=200, alpha=0.1):
    n_u, n_v = A.shape
    A_cal = A[cal_idx, :]

    deg_cal = A_cal.sum(axis=1) / n_v
    n_cal = len(deg_cal)

    # --- node-level bootstrap resampling ---
    deg_means = []
    for _ in range(n_boot):
        sample_idx = np.random.choice(n_cal, size=n_cal, replace=True)
        deg_mean = deg_cal[sample_idx].mean()
        deg_means.append(deg_mean)

    q_lo, q_hi = np.quantile(deg_means, [alpha/2, 1 - alpha/2])
    return q_lo, q_hi

# jacknife: Bootstrap은 랜덤 기반이고 Jackknife는 체계적인 leave-one-out 기반
def jackknife_ci(A, cal_idx, alpha=0.1):
    n_u, n_v = A.shape
    A_cal = A[cal_idx, :]
    n_cal = len(cal_idx)

    # 전체 평균 degree
    theta_hat = A_cal.sum(axis=1).mean() / n_v  

    # Jackknife leave-one-out 평균들
    theta_jack = []
    for i in range(n_cal):
        leave_out = np.delete(A_cal, i, axis=0)
        deg_leave = leave_out.sum(axis=1).mean() / n_v
        theta_jack.append(deg_leave)
    theta_jack = np.array(theta_jack)

    # Jackknife 분산 추정
    theta_bar = theta_jack.mean()
    var_hat = (n_cal - 1) / n_cal * np.sum((theta_jack - theta_bar) ** 2)

    # Gaussian CI
    se = np.sqrt(var_hat)
    z = st.norm.ppf(1 - alpha/2)

    q_lo = theta_hat - z * se
    q_hi = theta_hat + z * se

    return q_lo, q_hi

# subsampling: Without replacement / Node 대표성 불확실성 반영
def subsampling_ci(A, cal_idx, alpha=0.1, subsample_ratio=0.5, n_sub=200):
    n_u, n_v = A.shape
    sub_size = int(len(cal_idx) * subsample_ratio)
    deg_means = []

    # calibration subsampling
    for _ in range(n_sub):
        sub_idx = np.random.choice(cal_idx, size=sub_size, replace=False)
        deg_sub = A[sub_idx].sum(axis=1) / n_v
        deg_means.append(deg_sub.mean())

    # --- 2. subsample 분포에서 quantile 추정 ---
    deg_arr = np.array(deg_means)
    q_lo, q_hi = np.quantile(deg_arr, [alpha/2, 1 - alpha/2])

    return q_lo, q_hi

# Fast Patchwork Bootstrap (FPB)
# Paper: Bootstrap quantification of estimation uncertainties in network degree distributions
# patch degree 누적 → 분산 보존
# 패치를 resampling 단위로 쓰는 부트스트랩 방법 / 독립적인 샘플처럼 보이는 단위”를 새로 정의해야 함 → 그것이 patch
def fpb_ci(A, cal_idx, n_patch=50, waves=2, n_boot=200, alpha=0.1):
    n_u, n_v = A.shape
    A_sparse = csr_matrix(A)
    G = bipartite.from_biadjacency_matrix(A_sparse)

    deg_means = []

    for b in range(n_boot):
        sub_deg = np.zeros(n_u)
        counts = np.zeros(n_u)

        for _ in range(n_patch):
            seed = np.random.choice(cal_idx)

            # snowball sampling patch 생성
            patch_nodes = {seed}
            frontier = {seed}
            for _ in range(waves):
                new_frontier = set()
                for u in frontier:
                    neighbors = set(G.neighbors(u)) - patch_nodes
                    new_frontier |= neighbors
                patch_nodes |= new_frontier
                frontier = new_frontier
                if not frontier:
                    break

            # U쪽 노드만 반영
            for u in patch_nodes:
                if u < n_u:
                    sub_deg[u] += G.degree(u) / n_v
                    counts[u] += 1

        # 평균으로 보정
        counts[counts == 0] = 1
        sub_deg = sub_deg / counts

        # calibration subset의 평균 degree만 기록
        deg_mean = sub_deg[cal_idx].mean()
        deg_means.append(deg_mean)

    q_lo, q_hi = np.quantile(deg_means, [alpha/2, 1 - alpha/2])
    return q_lo, q_hi

# BootNet
# Paper: Bootstrap methods for various network estimation routines
# 데이터 전체로 네트워크 추정 후, bootstrap으로 불확실성 평가.
# post-hoc / 실제값이 bootstrap 추정 과정에 반영이 되어 있음
# bootnet (modified global interval baseline): 사실상 bootstrap과 동일
def bootnet_ci(A, cal_idx, n_boot=200, alpha=0.1):
    n_u, n_v = A.shape
    deg_cal = A[cal_idx].sum(axis=1) / n_v  # calibration degrees
    n_cal = len(cal_idx)

    # --- Bootstrap resampling ---
    ci_samples = []
    for _ in range(n_boot):
        sample_idx = resample(np.arange(n_cal), replace=True)
        deg_sample = deg_cal[sample_idx]
        ci_samples.append(deg_sample.mean())
    ci_samples = np.array(ci_samples)

    q_lo, q_hi = np.quantile(ci_samples, [alpha/2, 1 - alpha/2])

    return q_lo, q_hi

# ------------ Distributional ------------ #
# Gaussian Approximation
def gaussian_ci(A, cal_idx, alpha=0.1):
    n_u, n_v = A.shape
    A_cal = A[cal_idx, :]                 # calibration subset
    deg_cal = A_cal.sum(axis=1) / n_v     # calibration degrees
    theta_hat = deg_cal.mean()           
    
    se = deg_cal.std(ddof=1) / np.sqrt(len(deg_cal)) 
    z = st.norm.ppf(1 - alpha/2)         
    
    q_lo = theta_hat - z * se
    q_hi = theta_hat + z * se
    return q_lo, q_hi

# Bayesian
# cross-validation 식 평가: train으로 posterior 학습 → test로 coverage 평가
# Bernoulli 확률 추정: Bayesian credible interval (Beta-Binomial 모델 기반)
def bayesian_ci(A, cal_idx, test_idx, alpha=0.1, draws=2000, tune=1000, chains=1, cores=1):
    n_u, n_v = A.shape

    # Calibration data: edges of calibration nodes (flatten to 0/1)
    train_data = A[cal_idx].flatten().astype(int)

    with pm.Model() as model:
        theta = pm.Beta("theta", alpha=1, beta=1)
        y_obs = pm.Bernoulli("y_obs", p=theta, observed=train_data)
        trace = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=cores,
            target_accept=1-alpha, init="adapt_diag",
            progressbar=True
        )

    theta_samples = trace.posterior["theta"].values.flatten()  # (n_samples,)

    # --- Posterior predictive for test nodes ---
    lower, upper = [], []
    for _ in test_idx:
        test_samples = np.random.binomial(n=n_v, p=theta_samples) / n_v
        q_lo, q_hi = np.quantile(test_samples, [alpha/2, 1 - alpha/2])
        lower.append(q_lo)
        upper.append(q_hi)

    return np.array(lower), np.array(upper)

# Variational Bayesian
class VBNetworkCentrality(nn.Module):
    def __init__(self, n_nodes):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(n_nodes))
        self.log_sigma = nn.Parameter(torch.zeros(n_nodes))

    def forward(self, edges, n_samples=1):
        mu, log_sigma = self.mu, self.log_sigma
        sigma = torch.exp(log_sigma)
        eps = torch.randn(n_samples, len(mu))
        c_samples = mu + sigma * eps  # (n_samples, n_nodes)

        loglik = 0
        for (i,j) in edges:
            ci, cj = c_samples[:, i], c_samples[:, j]
            p = torch.sigmoid(ci * cj)
            loglik += torch.log(p + 1e-8).mean()
        return loglik

def vb_ci(A, cal_idx, test_idx, n_iter=2000, lr=1e-2, n_samples=10, alpha=0.1):
    n_u, n_v = A.shape
    n = n_u + n_v   # total nodes in bipartite graph

    model = VBNetworkCentrality(n)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Build train/test edge lists ---
    # calibration edges (U nodes restricted to cal_idx)
    train_edges = []
    for u in cal_idx:
        for v in range(n_v):
            if A[u, v] == 1:
                train_edges.append((u, n_u + v))  # shift v index

    # test edges (U nodes restricted to test_idx)
    test_edges = []
    for u in test_idx:
        for v in range(n_v):
            if A[u, v] == 1:
                test_edges.append((u, n_u + v))

    # --- Training with ELBO ---
    for step in range(n_iter):
        optimizer.zero_grad()
        loglik = model(train_edges, n_samples=n_samples)
        mu, log_sigma = model.mu, model.log_sigma
        kl = -0.5 * torch.sum(1 + 2*log_sigma - mu**2 - torch.exp(2*log_sigma))
        loss = -(loglik - kl)
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"Iter {step}, ELBO: {-loss.item():.4f}")

    # --- Test prediction ---
    mu, sigma = model.mu.detach().numpy(), np.exp(model.log_sigma.detach().numpy())
    q_lo_lst, q_hi_lst = [], []

    for (i, j) in test_edges:
        ci_samples = np.random.normal(mu[i], sigma[i], size=1000)
        cj_samples = np.random.normal(mu[j], sigma[j], size=1000)
        p_samples = expit(ci_samples * cj_samples)
        q_lo, q_hi = np.quantile(p_samples, [alpha/2, 1-alpha/2])
        q_lo_lst.append(q_lo)
        q_hi_lst.append(q_hi)

    return np.array(q_lo_lst), np.array(q_hi_lst)

# MC dropout 
# 간단한 MLP with Dropout
class MLPDropout(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        return torch.sigmoid(self.fc2(h))

def mcdropout_ci(model, train_data, test_data, n_forward=200, alpha=0.1, lr=1e-2, n_iter=1000):
    X_train, y_train = train_data
    X_test = test_data

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()

    # --- Training ---
    model.train()
    for step in range(n_iter):
        optimizer.zero_grad()
        pred = model(X_train).squeeze()
        loss = loss_fn(pred, y_train.float())
        loss.backward()
        optimizer.step()
        if step % 200 == 0:
            print(f"Step {step}, Loss {loss.item():.4f}")

    # --- Test Prediction with MC Dropout ---
    preds = []
    model.train()  # Dropout 활성화 유지
    with torch.no_grad():
        for _ in range(n_forward):
            pred = model(X_test).squeeze().cpu().numpy()
            preds.append(pred)
    preds = np.array(preds)  # (n_forward, n_test)

    # 분위수 기반 CI
    q_lo = np.quantile(preds, alpha/2, axis=0)
    q_hi = np.quantile(preds, 1 - alpha/2, axis=0)

    return q_lo, q_hi

# ------------ Predictive ------------ #
# Conformal Prediction
def cp_ci(deg_u, cal_idx, test_idx, alpha=0.1):

    # "예측"은 surrogate가 없으니 그냥 전체 평균 degree 사용
    baseline_pred = np.repeat(deg_u[cal_idx].mean(), len(cal_idx))

    # Calibration residuals
    scores = np.abs(deg_u[cal_idx] - baseline_pred)

    # Quantile cutoff
    qhat = np.quantile(scores, 1 - alpha)

    # Test set: 동일 baseline 사용
    test_pred = np.repeat(deg_u[test_idx].mean(), len(test_idx))
    lower_test = test_pred - qhat
    upper_test = test_pred + qhat

    return lower_test, upper_test

# Predictive Model
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, adj_norm):
        h = torch.spmm(adj_norm, x)
        return self.fc(h)
    
class GCNRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, 1)

    def forward(self, x, adj_norm):
        h = F.relu(self.gcn1(x, adj_norm))
        out = self.gcn2(h, adj_norm)
        return out.squeeze()

def gcn_ci(A, deg_u, train_idx, test_idx, alpha=0.1, hidden_dim=64, lr=1e-2, n_epochs=200):
    n_u, n_v = A.shape
    # Bipartite user-item adjacency → user-user adjacency (projection)
    A_user = A @ A.T
    A_user = A_user + np.eye(n_u)  # self-loop

    # Normalize adjacency
    D = np.diag(1.0 / np.sqrt(A_user.sum(axis=1)))
    A_norm = D @ A_user @ D

    # Convert to Torch tensors
    X = torch.eye(n_u)                   # one-hot user features
    adj_norm = torch.tensor(A_norm, dtype=torch.float32)
    y = torch.tensor(deg_u, dtype=torch.float32)

    # Model
    model = GCNRegressor(in_dim=n_u, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # --- Training ---
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X, adj_norm)
        loss = loss_fn(y_pred[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.4f}")

    # --- Prediction ---
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X, adj_norm)[train_idx].numpy()
        y_pred_test = model(X, adj_norm)[test_idx].numpy()

    # --- Residuals from training ---
    residuals = np.abs(deg_u[train_idx] - y_pred_train)
    qhat = np.quantile(residuals, 1 - alpha)

    # --- CI for test nodes ---
    lower_test = y_pred_test - qhat
    upper_test = y_pred_test + qhat

    return lower_test, upper_test

def lr_ci(A, deg_u, train_idx, test_idx, alpha=0.1):
    # Features: item connection vectors (rows of A)
    X_train, y_train = A[train_idx], deg_u[train_idx]
    X_test, y_test = A[test_idx], deg_u[test_idx]

    # --- Simple Linear Regression ---
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Residuals (estimate error distribution from training set)
    residuals = np.abs(y_train - y_pred_train)
    qhat = np.quantile(residuals, 1 - alpha)

    # Prediction intervals
    lower_test = y_pred_test - qhat
    upper_test = y_pred_test + qhat

    return lower_test, upper_test

# ------------ Graphon ------------ #
# Paper: Graphon Estimation in Bipartite Graphs with Observable Edge Labels and Unobservable Node Labels
# “latent position by degree sort” → 논문에서는 latent 위치를 알려주지 않고, 통계적 방법으로 암묵적 재배치(reordering)나 partition/clustering 기법을 씀. 너 코드의 간단한 degree-sorting은 논문 방식보다는 단순한 heuristics임.
def graphon_ci(A, deg_u, cal_idx, test_idx, alpha=0.1, bandwidth=0.1, method="cp"):
    n_u, n_v = A.shape

    # --- Step 1: Latent position by degree sorting ---
    sorted_idx = np.argsort(deg_u)
    latent_pos = np.linspace(0, 1, n_u)
    x_map = np.zeros(n_u)
    x_map[sorted_idx] = latent_pos  # assign latent positions by rank

    x_cal = x_map[cal_idx].reshape(-1, 1)
    y_cal = deg_u[cal_idx]

    x_test = x_map[test_idx].reshape(-1, 1)

    # --- Step 2: Prediction function (kernel smoother baseline) ---
    def kernel(u): return np.exp(-u**2 / (2 * bandwidth**2))

    def predict(x_target, x_ref, y_ref):
        preds = []
        for x in x_target:
            weights = kernel(np.abs(x_ref - x))
            preds.append(np.sum(weights * y_ref) / np.sum(weights))
        return np.array(preds)

    # Calibration predictions
    y_cal_pred = predict(x_cal, x_cal, y_cal)

    # Test predictions
    y_test_pred = predict(x_test, x_cal, y_cal)

    # --- Step 3: Interval construction ---
    if method == "gaussian":
        # 기존 Gaussian 방식
        resid = y_cal - y_cal_pred
        sigma_hat = resid.std()
        z = st.norm.ppf(1 - alpha/2)
        half_width = z * sigma_hat
        lower, upper = y_test_pred - half_width, y_test_pred + half_width

    elif method == "qr":
        # Quantile Regression residuals
        qr_lo = QuantileRegressor(quantile=alpha/2, solver="highs").fit(x_cal, y_cal)
        qr_hi = QuantileRegressor(quantile=1 - alpha/2, solver="highs").fit(x_cal, y_cal)
        lower = qr_lo.predict(x_test)
        upper = qr_hi.predict(x_test)

    elif method == "cp":
        # Conformal Prediction with residual quantile
        resid = np.abs(y_cal - y_cal_pred)
        qhat = np.quantile(resid, 1 - alpha)
        lower, upper = y_test_pred - qhat, y_test_pred + qhat

    else:
        raise ValueError("method must be one of ['gaussian', 'qr', 'cp']")

    return lower, upper

# ------------ Graphex ------------ #
# Graphex-based Conformal Prediction
def graphex_cp_ci_bf(A, deg_u, prob_matrix, alpha=0.1, test_size=0.3, seed=42,
    adaptive=True, var_mode="binomial", score_type="absolute"
    ):
    
    n_u = len(deg_u)
    idx = np.arange(n_u)

    # Calibration/Test split
    cal_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed)

    # Expected degree (mean over V side)
    expected_deg = prob_matrix.mean(axis=1)

    # Nonconformity score
    if score_type == "absolute":
        scores = np.abs(deg_u[cal_idx] - expected_deg[cal_idx])
    elif score_type == "relative":
        scores = np.abs(deg_u[cal_idx] - expected_deg[cal_idx]) / (expected_deg[cal_idx] + 1e-6)
    elif score_type == "squared":
        scores = (deg_u[cal_idx] - expected_deg[cal_idx])**2
    elif score_type == "weighted":
        weights = np.log1p(expected_deg[cal_idx]) 
        scores = weights * np.abs(deg_u[cal_idx] - expected_deg[cal_idx])
    else:
        raise ValueError("score_type must be one of {'absolute','relative','squared','weighted'}")

    # Quantile threshold
    scores_sorted = np.sort(scores)
    trim = int(0.05 * len(scores_sorted))   # top 5% trim
    qhat = np.quantile(scores_sorted[:-trim], 1 - alpha)

    # Variance-aware scaling 
    if adaptive:
        if var_mode == "binomial":
            n_v = A.shape[1]
            var_est = expected_deg * (1 - expected_deg / n_v)
        elif var_mode == "full":
            var_est = (prob_matrix * (1 - prob_matrix)).sum(axis=1)
        else:
            raise ValueError("var_mode must be 'binomial' or 'full'")
        scale = np.sqrt(var_est[test_idx] + 1e-6) ** 0.5 
    else:
        scale = 1.0

    # Prediction intervals
    # lower_test = expected_deg[test_idx] - qhat * scale
    # upper_test = expected_deg[test_idx] + qhat * scale
    lower_test = expected_deg[test_idx] - (qhat + 0.5*scale)  # additive hybrid
    upper_test = expected_deg[test_idx] + (qhat + 0.5*scale)

    return cal_idx, test_idx, lower_test, upper_test, expected_deg[test_idx]

def graphex_cp_ci(A, deg_u, cal_idx, test_idx, prob_matrix, alpha=0.1,
    score_type="absolute", finite_correction=True, trim=0.0, 
    adaptive=True, var_mode="binomial", gamma=0.01, delta=0.05
    ):

    expected_deg = prob_matrix.mean(axis=1)

    if score_type == "absolute":
        scores = np.abs(deg_u[cal_idx] - expected_deg[cal_idx])
    elif score_type == "relative":
        scores = np.abs(deg_u[cal_idx] - expected_deg[cal_idx]) / (expected_deg[cal_idx] + 1e-6)
    elif score_type == "squared":
        scores = (deg_u[cal_idx] - expected_deg[cal_idx])**2
    elif score_type == "weighted":
        weights = np.log1p(expected_deg[cal_idx])
        scores = weights * np.abs(deg_u[cal_idx] - expected_deg[cal_idx])
    else:
        raise ValueError("score_type must be one of {'absolute','relative','squared','weighted'}")

    if trim > 0:
        scores_sorted = np.sort(scores)
        trim_n = int(trim * len(scores_sorted))
        scores_sorted = scores_sorted[:-trim_n] if trim_n > 0 else scores_sorted
    else:
        scores_sorted = scores

    if finite_correction:
        q_level = np.ceil((len(scores_sorted)+1) * (1-alpha)) / len(scores_sorted)
        q_level = min(1.0, q_level + 1/len(scores_sorted))
    else:
        q_level = 1 - alpha

    qhat = np.quantile(scores_sorted, q_level, method="higher")
    qhat *= (1.0 + delta)   # conservative buffer

    if adaptive:
        if var_mode == "binomial":
            n_v = A.shape[1]
            var_est = expected_deg * (1 - expected_deg / n_v)
        elif var_mode == "full":
            var_est = (prob_matrix * (1 - prob_matrix)).sum(axis=1)
        else:
            raise ValueError("var_mode must be 'binomial' or 'full'")
        
        # (1) normalize by mean variance (relative scale)
        var_est = var_est / (np.mean(var_est) + 1e-6)

        # (2) softer gamma (default: 0.25 instead of 1.0)
        scale = (var_est[test_idx] + 1e-6) ** (gamma / 2)

        # (3) cap maximum scaling factor (e.g., 3.0)
        scale = np.minimum(scale, 3.0)
    else:
        scale = 1.0

    lower_test = expected_deg[test_idx] - qhat * scale
    upper_test = expected_deg[test_idx] + qhat * scale

    return lower_test, upper_test
