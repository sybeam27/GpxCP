import numpy as np
import time
import networkx as nx
from networkx.algorithms import bipartite
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.data import build_adjacency, graph_density, build_bipartite_graph, load_dataset
from utils.graphex import generate_bipartite_graphex
from utils.evaluate import check_sparsity, evaluate_ci_extended

from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
import numpy as np
import networkx as nx

test_size = 0.3
seed = 42

alpha = 0.1
data ='real' 
T_lst = [100, 500, 1000, 5000]

# Dataset
if data == 'syn':
    # synthetic data: graphex
    A_dict, prob_matrix_dict = {}, {}
    
    for t in T_lst:
        A_syn, U, V, comp = generate_bipartite_graphex(T=t, alpha=0.5, p_star=0.05, p_noise=0.01, seed=123)
        prob_matrix_syn = comp["W_total"]
        print('=' * 50)
        print("Synthetic data Adjacency shape:", A_syn.shape)
        check_sparsity(A_syn)
        graph_density(A_syn, threshold=0.1)
        
        A_dict[f'Syn_{t}'] = A_syn
        prob_matrix_dict[f'Syn_{t}'] = prob_matrix_syn
    
elif data == 'real':
    # MovieLens - 100K
    print("=" * 10, 'MovieLens - 100K', '=' * 10)
    mv_df = load_dataset("ml-100k")
    # print(mv_df.head())

    B = build_bipartite_graph(mv_df, user_col="user_id", item_col="item_id")
    print(f"총 노드 수: {B.number_of_nodes()}, 총 엣지 수: {B.number_of_edges()}")

    A_mv = build_adjacency(mv_df, threshold=3)
    check_sparsity(A_mv)
    print('=' * 50)
    print("MovieLense-100K Adjacency matrix shape:", A_mv.shape) 
    graph_density(A_mv, threshold=0.1)

    # MovieLense - 1M
    print("=" * 10, 'MovieLense - 1M', '=' * 10)
    mv2_df = load_dataset("ml-1m")
    # print(mv2_df.head())

    B = build_bipartite_graph(mv2_df, user_col="user_id", item_col="item_id")
    print(f"총 노드 수: {B.number_of_nodes()}, 총 엣지 수: {B.number_of_edges()}")

    A_mv2 = build_adjacency(mv2_df, threshold=3)
    check_sparsity(A_mv2)
    print("="*50)
    print("MovieLense-1M Adjacency matrix shape:", A_mv2.shape)
    graph_density(A_mv2, threshold=0.1)

    # CiteULike 데이터셋
    print("=" * 10, 'CiteULike', '=' * 10)
    cl_df = load_dataset("citeulike")
    # print(cl_df.head())

    B = build_bipartite_graph(cl_df, user_col="user_id", item_col="item_id")
    print(f"총 노드 수: {B.number_of_nodes()}, 총 엣지 수: {B.number_of_edges()}")

    A_cite = build_adjacency(cl_df, threshold=0)
    check_sparsity(A_cite)
    print("="*50)
    print("CiteULike Adjacency matrix shape:", A_cite.shape)
    graph_density(A_cite, threshold=0.1)

    # DBLP 데이터셋
    print("=" * 10, 'DBLP', '=' * 10)
    dblp_df = load_dataset("dblp")
    # print(dblp_df.head())

    B = build_bipartite_graph(dblp_df, user_col="user_id", item_col="item_id")
    print(f"총 노드 수: {B.number_of_nodes()}, 총 엣지 수: {B.number_of_edges()}")

    A_dblp = build_adjacency(dblp_df, threshold=0)
    check_sparsity(A_dblp)
    print("="*50)
    print("DBLP Adjacency matrix shape:", A_dblp.shape)
    graph_density(A_dblp, threshold=0.1)

    # Wiki 데이터셋
    print("=" * 10, 'Wiki', '=' * 10)
    wiki_df = load_dataset("wiki")
    # print(wiki_df.head())

    B = build_bipartite_graph(wiki_df, user_col="user_id", item_col="item_id")
    print(f"총 노드 수: {B.number_of_nodes()}, 총 엣지 수: {B.number_of_edges()}")

    A_wiki = build_adjacency(wiki_df, threshold=0)
    check_sparsity(A_wiki)
    print("="*50)
    print("Wiki adjacency matrix shape:", A_wiki.shape)
    graph_density(A_wiki, threshold=0.1)

    A_dict = {
        'movie_k': A_mv,
        'movie_m': A_mv2,
        'cite': A_cite,
        'dblp': A_dblp,
        'wiki': A_wiki
    }

def estimate_graphex_prob(A, centrality_vals=None, alpha=0.1, p_star=0.05, p_noise=0.01):
    """
    Graphex surrogate generalized to any centrality measure.
    - centrality_vals: node-level scores (degree, eigenvector, betweenness, closeness 등)
    """
    n = A.shape[0]

    if centrality_vals is None:
        # 기본값: degree 중심성
        deg = A.sum(axis=1) / n
        centrality_vals = deg
    
    # 정규화
    c = centrality_vals / (centrality_vals.max() + 1e-8)

    # Kernel from centrality outer product
    W_kernel = np.outer(c, c) * alpha

    # Star hubs (상위 10%)
    W_star = np.zeros_like(W_kernel)
    hubs = np.argsort(-c)[:max(1, n//10)]
    W_star[hubs, :] += p_star
    W_star[:, hubs] += p_star

    # Noise
    W_noise = np.full_like(W_kernel, p_noise)

    return np.clip(W_kernel + W_star + W_noise, 0, 1)

def graphex_cp_ci(A, cent_vals, cal_idx, test_idx, prob_matrix, alpha=0.1,
    score_type="absolute", finite_correction=True, trim=0.0, 
    adaptive=False, var_mode="binomial", gamma=1.0):
    """
    Graphex-CP interval estimation generalized to arbitrary centrality values.
    cent_vals: node-level centrality values
    """
    expected_vals = prob_matrix.mean(axis=1)

    # Nonconformity score
    if score_type == "absolute":
        scores = np.abs(cent_vals[cal_idx] - expected_vals[cal_idx])
    elif score_type == "relative":
        scores = np.abs(cent_vals[cal_idx] - expected_vals[cal_idx]) / (expected_vals[cal_idx] + 1e-6)
    elif score_type == "squared":
        scores = (cent_vals[cal_idx] - expected_vals[cal_idx])**2
    elif score_type == "weighted":
        weights = np.log1p(expected_vals[cal_idx])
        scores = weights * np.abs(cent_vals[cal_idx] - expected_vals[cal_idx])
    else:
        raise ValueError("score_type must be one of {'absolute','relative','squared','weighted'}")

    # Quantile threshold
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
    qhat *= 1.05   # conservative buffer

    # Variance-aware scaling
    if adaptive:
        if var_mode == "binomial":
            n_v = A.shape[1]
            var_est = expected_vals * (1 - expected_vals / n_v)
        elif var_mode == "full":
            var_est = (prob_matrix * (1 - prob_matrix)).sum(axis=1)
        else:
            raise ValueError("var_mode must be 'binomial' or 'full'")
        # γ-exponent scaling
        scale = (var_est[test_idx] + 1e-6) ** (gamma/2)
    else:
        scale = 1.0

    # Prediction intervals
    lower_test = expected_vals[test_idx] - qhat * scale
    upper_test = expected_vals[test_idx] + qhat * scale

    return lower_test, upper_test

# 네트워크가 연결 그래프가 아니라서 nx.eigenvector_centrality_numpy가 실패
# 컴포넌트 단위로 eigenvector_centrality 적용
def safe_eigenvector_centrality(G, U_nodes):
    """
    Eigenvector centrality 계산 (disconnected graph 대응).
    각 connected component 별로 계산 후 결과 병합.
    """
    centrality = {}
    for comp in nx.connected_components(G):
        subG = G.subgraph(comp)
        try:
            c = nx.eigenvector_centrality(subG, max_iter=500, tol=1e-06)
            centrality.update(c)
        except nx.NetworkXException:
            # 실패하면 0으로 채움
            for u in comp:
                centrality[u] = 0.0
    return np.array([centrality[u] for u in U_nodes])


def fast_degree(A):
    return A.sum(axis=1) / A.shape[0]

def fast_eigenvector_bipartite(A, mode="user"):
    if mode == "user":
        M = A @ A.T   # user-user projection
    else:
        M = A.T @ A   # item-item projection
    A_sparse = csr_matrix(M)
    vals, vecs = eigs(A_sparse, k=1, which='LR')
    v = np.abs(vecs[:,0].real)
    return v / v.sum()

def fast_closeness(G, sample_size=500):
    # 샘플링 기반 closeness
    import random
    nodes = list(G.nodes())
    sample_nodes = random.sample(nodes, min(sample_size, len(nodes)))
    closeness = {u:0 for u in nodes}
    for s in sample_nodes:
        lengths = nx.single_source_shortest_path_length(G, s)
        for u, d in lengths.items():
            closeness[u] += d
    # 평균 거리 기반으로 근사 closeness 계산
    n = len(sample_nodes)
    for u in closeness:
        if closeness[u] > 0:
            closeness[u] = (n-1) / closeness[u]
        else:
            closeness[u] = 0
    return np.array([closeness[u] for u in range(len(closeness))])

def fast_betweenness(G, k=500):
    # 샘플링 기반 betweenness
    btw = nx.betweenness_centrality(G, k=k, seed=42)
    return np.array([btw[u] for u in range(len(btw))])


centrality_methods = {
    "degree": lambda A, G, U_nodes: fast_degree(A),
    "eigenvector": lambda A, G, U_nodes: fast_eigenvector_bipartite(A),
    "closeness": lambda A, G, U_nodes: fast_closeness(G),
    "betweenness": lambda A, G, U_nodes: fast_betweenness(G),
}

def build_bipartite_graph_from_A(A):
    n_u, n_v = A.shape
    G = nx.Graph()
    # U 파티션 (0 ~ n_u-1), V 파티션 (n_u ~ n_u+n_v-1)
    G.add_nodes_from(range(n_u), bipartite=0)
    G.add_nodes_from(range(n_u, n_u+n_v), bipartite=1)
    # 간선 추가
    for i in range(n_u):
        js = np.where(A[i] > 0)[0]
        for j in js:
            G.add_edge(i, n_u + j)
    return G, list(range(n_u))  # U 파티션 노드 반환

for key, A in tqdm(A_dict.items(), desc="Datasets"):
    print(f"\n===== Dataset: {key} =====")
    G, U_nodes = build_bipartite_graph_from_A(A)

    for cname, cfunc in centrality_methods.items():
        if cname == 'degree':
            n = A.shape[0]
            deg = A.sum(axis=1) / n
            centrality_vals = deg
            prob_matrix = estimate_graphex_prob(A, centrality_vals=deg)
            cal_idx, test_idx = train_test_split(np.arange(n), test_size=test_size, random_state=seed)
            lower, upper = graphex_cp_ci(A, deg, cal_idx, test_idx, prob_matrix=prob_matrix, alpha=alpha)
            res = evaluate_ci_extended(deg[test_idx], lower, upper, alpha=alpha)
        else:
            # 중심성 값 (U 파티션만)
            cent_vals = cfunc(A, G, U_nodes)
            prob_matrix = estimate_graphex_prob(A, centrality_vals=cent_vals)
            cal_idx, test_idx = train_test_split(np.arange(len(cent_vals)), test_size=test_size, random_state=seed)
            lower, upper = graphex_cp_ci(A, cent_vals, cal_idx, test_idx, prob_matrix=prob_matrix, alpha=alpha)
            res = evaluate_ci_extended(cent_vals[test_idx], lower, upper, alpha=alpha)


        print(f"[{cname:10s}] Coverage={res['coverage']:.4f}, "
              f"Width={res['avg_width']:.4f}, "
              f"IScore={res['interval_score']:.4f}")

