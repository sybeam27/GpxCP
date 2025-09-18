# exp_baseline.py

import os
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.data import build_adjacency, graph_density, build_bipartite_graph, load_dataset
from utils.graphex import estimate_graphex_prob, generate_bipartite_graphex
from utils.evaluate import check_sparsity, evaluate_ci_extended

from utils.model import (
    bootstrap_ci, subsampling_ci, bootnet_ci,
    gaussian_ci, bayesian_ci,
    cp_ci, lr_ci, graphon_ci, graphex_cp_ci
)

# === Global Parameters ===
test_size = 0.3
seed = 42
lr = 1e-2

n_boot = 200
r_sub = 0.5
n_sub = 200
n_draw = 500
n_tune = 1000
n_chain = 4   # 윈도우 multiprocessing 문제 방지: 1
n_iter = 2000
n_sample = 10
bandwidth = 0.05

alpha = 0.1
data = 'real'   # or 'syn'
T_lst = [100, 500, 1000, 5000]

def prepare_datasets():
    if data == 'syn':
        A_dict, prob_matrix_dict = {}, {}
        for t in T_lst:
            A_syn, U, V, comp = generate_bipartite_graphex(
                T=t, alpha=0.5, p_star=0.05, p_noise=0.01, seed=123
            )
            prob_matrix_syn = comp["W_total"]
            print("=" * 50)
            print("Synthetic data Adjacency shape:", A_syn.shape)
            check_sparsity(A_syn)
            graph_density(A_syn, threshold=0.1)

            A_dict[f'Syn_{t}'] = A_syn
            prob_matrix_dict[f'Syn_{t}'] = prob_matrix_syn
        return A_dict, prob_matrix_dict

    elif data == 'real':
        A_dict, prob_matrix_dict = {}, {}

        # MovieLens - 100K
        print("=" * 10, 'MovieLens - 100K', '=' * 10)
        mv_df = load_dataset("ml-100k")
        B = build_bipartite_graph(mv_df, user_col="user_id", item_col="item_id")
        print(f"총 노드 수: {B.number_of_nodes()}, 총 엣지 수: {B.number_of_edges()}")

        A_mv = build_adjacency(mv_df, threshold=3)
        check_sparsity(A_mv)
        prob_matrix_mv = estimate_graphex_prob(A_mv)
        print("=" * 50)
        print("MovieLens-100K Adjacency matrix shape:", A_mv.shape)
        graph_density(A_mv, threshold=0.1)

        # MovieLens - 1M
        print("=" * 10, 'MovieLens - 1M', '=' * 10)
        mv2_df = load_dataset("ml-1m")
        B = build_bipartite_graph(mv2_df, user_col="user_id", item_col="item_id")
        print(f"총 노드 수: {B.number_of_nodes()}, 총 엣지 수: {B.number_of_edges()}")

        A_mv2 = build_adjacency(mv2_df, threshold=3)
        check_sparsity(A_mv2)
        prob_matrix_mv2 = estimate_graphex_prob(A_mv2)
        print("=" * 50)
        print("MovieLens-1M Adjacency matrix shape:", A_mv2.shape)
        graph_density(A_mv2, threshold=0.1)

        # CiteULike
        print("=" * 10, 'CiteULike', '=' * 10)
        cl_df = load_dataset("citeulike")
        B = build_bipartite_graph(cl_df, user_col="user_id", item_col="item_id")
        print(f"총 노드 수: {B.number_of_nodes()}, 총 엣지 수: {B.number_of_edges()}")

        A_cite = build_adjacency(cl_df, threshold=0)
        check_sparsity(A_cite)
        prob_matrix_cite = estimate_graphex_prob(A_cite)
        print("=" * 50)
        print("CiteULike Adjacency matrix shape:", A_cite.shape)
        graph_density(A_cite, threshold=0.1)

        # DBLP
        print("=" * 10, 'DBLP', '=' * 10)
        dblp_df = load_dataset("dblp")
        B = build_bipartite_graph(dblp_df, user_col="user_id", item_col="item_id")
        print(f"총 노드 수: {B.number_of_nodes()}, 총 엣지 수: {B.number_of_edges()}")

        A_dblp = build_adjacency(dblp_df, threshold=0)
        check_sparsity(A_dblp)
        prob_matrix_dblp = estimate_graphex_prob(A_dblp)
        print("=" * 50)
        print("DBLP Adjacency matrix shape:", A_dblp.shape)
        graph_density(A_dblp, threshold=0.1)

        # Wiki
        print("=" * 10, 'Wiki', '=' * 10)
        wiki_df = load_dataset("wiki")
        B = build_bipartite_graph(wiki_df, user_col="user_id", item_col="item_id")
        print(f"총 노드 수: {B.number_of_nodes()}, 총 엣지 수: {B.number_of_edges()}")

        A_wiki = build_adjacency(wiki_df, threshold=0)
        check_sparsity(A_wiki)
        prob_matrix_wiki = estimate_graphex_prob(A_wiki)
        print("=" * 50)
        print("Wiki Adjacency matrix shape:", A_wiki.shape)
        graph_density(A_wiki, threshold=0.1)

        A_dict = {
            'movie_k': A_mv,
            'movie_m': A_mv2,
            'cite': A_cite,
            'dblp': A_dblp,
            'wiki': A_wiki,
        }

        prob_matrix_dict = {
            'movie_k': prob_matrix_mv,
            'movie_m': prob_matrix_mv2,
            'cite': prob_matrix_cite,
            'dblp': prob_matrix_dblp,
            'wiki': prob_matrix_wiki,
        }

        return A_dict, prob_matrix_dict

def run_experiments(A_dict):
    os.makedirs(".results", exist_ok=True)

    for key in tqdm(A_dict, desc="Datasets"):
        A = A_dict[key]
        n_u, n_v = A.shape
        deg_u = A.sum(axis=1) / n_v
        cal_idx, test_idx = train_test_split(np.arange(n_u), test_size=test_size, random_state=seed)

        baselines, timings, results_all = {}, {}, {}

        methods = [
            ("Bootstrap", lambda: bootstrap_ci(A, cal_idx, n_boot=n_boot, alpha=alpha)),
            ("Subsampling", lambda: subsampling_ci(A, cal_idx, alpha=alpha, subsample_ratio=r_sub, n_sub=n_sub)),
            ("BootNet", lambda: bootnet_ci(A, cal_idx, n_boot=n_boot, alpha=alpha)),
            ("Gaussian", lambda: gaussian_ci(A, cal_idx, alpha=alpha)),
            ("Bayesian", lambda: bayesian_ci(A, cal_idx, test_idx, alpha=alpha, draws=n_draw, tune=n_tune, chains=n_chain)),
            ("CP", lambda: cp_ci(deg_u, cal_idx, test_idx, alpha=alpha)),
            ("LR", lambda: lr_ci(A, deg_u, cal_idx, test_idx, alpha=alpha)),
            ("Graphon", lambda: graphon_ci(A, deg_u, cal_idx, test_idx, alpha=alpha, bandwidth=bandwidth, method="cp")),
            ("Graphex", lambda: graphex_cp_ci(
                A, deg_u, cal_idx, test_idx,
                prob_matrix=estimate_graphex_prob(A), alpha=alpha,
                score_type="absolute", finite_correction=True,
                trim=0.0, adaptive=True, var_mode="binomial",
                gamma=0.01, delta=0.05
            )),
        ]

        for mname, func in tqdm(methods, desc=f"{key} Methods", leave=False):
            # print(f"{key} Methods Start....")
            start = time.time()
            lower, upper = func()
            timings[mname] = time.time() - start
            baselines[mname] = (lower, upper)

        # --- Evaluation ---
        print(f"\n=== 성능 ({key}, Test only, alpha={alpha}) ===")
        print(f"{'Method':12s} | {'Cov':>6s} | {'Width':>8s} | {'NIW':>8s} | {'IScore':>8s} | {'Time(s)':>8s}")
        print("-" * 70)

        for mname, (lower, upper) in baselines.items():
            res = evaluate_ci_extended(deg_u[test_idx], lower, upper, alpha=alpha)
            res["time"] = timings[mname]
            res["alpha"] = alpha
            res["dataset"] = key
            results_all[mname] = res

            print(f"{mname:12s} | {res['coverage']:6.4f} | {res['avg_width']:6.4f}"
                  f"{res['niw']:8.4f} | {res['interval_score']:8.4f} | "
                  f"{res['time']:8.4f}")

        # Save results to CSV
        df = pd.DataFrame.from_dict(results_all, orient="index")
        df.index.name = "method"
        out_path = f".results/results_{key}.csv"
        df.to_csv(out_path)
        print(f"[INFO] Saved results to {out_path}")

def main():
    A_dict, _ = prepare_datasets()
    run_experiments(A_dict)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
