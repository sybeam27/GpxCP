import os
import zipfile
import numpy as np
import pandas as pd
import networkx as nx
from cornac.datasets import citeulike

def unzip_file(zip_path, extract_to="."):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"{zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Complete: {extract_to}")

def build_adjacency(ratings, threshold = 1):
    users = ratings["user_id"].unique()
    items = ratings["item_id"].unique()

    user_list = sorted(users)
    item_list = sorted(items)

    user_to_idx = {u: idx for idx, u in enumerate(user_list)}
    item_to_idx = {i: idx for idx, i in enumerate(item_list)}

    A = np.zeros((len(user_list), len(item_list)), dtype=int)
    for row in ratings.itertuples():
        u = user_to_idx[row.user_id]
        v = item_to_idx[row.item_id]
        A[u, v] = 1 if row.rating >= threshold else 0
    return A

def graph_density(A, threshold=0.05, verbose=True):
    n_u, n_v = A.shape
    num_edges = A.sum()
    max_edges = n_u * n_v
    density = num_edges / max_edges
    is_sparse = density < threshold
    
    if verbose:
        print(f"Number of nodes (U,V): ({n_u}, {n_v})")
        print(f"Number of edges: {int(num_edges)} / Max edges: {max_edges}")
        print(f"Density = {density:.4f}")
        print(f"Sparse: {'Yes' if is_sparse else 'No'} (threshold={threshold})")
    
    # return density, is_sparse

def build_bipartite_graph(df, user_col="user_id", item_col="item_id"):
    B = nx.Graph()
    
    # Unique nodes
    users = df[user_col].unique()
    items = df[item_col].unique()
    
    # Add nodes with bipartite attribute
    B.add_nodes_from([f"u_{u}" for u in users], bipartite=0)
    B.add_nodes_from([f"i_{i}" for i in items], bipartite=1)
    
    # Add edges
    edges = [(f"u_{row._asdict()[user_col]}", f"i_{row._asdict()[item_col]}")
             for row in df.itertuples()]
    B.add_edges_from(edges)
    
    return B

def load_dataset(name, base_path="./data"):
    """
    Load different benchmark recommendation datasets into a DataFrame.
    
    Supported datasets:
    - "ml-100k": MovieLens 100K
    - "ml-1m"  : MovieLens 1M
    - "citeulike": CiteULike feedback dataset
    - "dblp"   : DBLP rating dataset
    - "wiki"   : Wiki dataset
    
    Parameters
    ----------
    name : str
        Dataset name (one of {"ml-100k", "ml-1m", "citeulike", "dblp", "wiki"}).
    base_path : str, optional
        Base folder path where datasets are stored (default="data").
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns at least ["user_id", "item_id", "rating"].
        Additional columns (e.g., "title") may be included if available.
    """
    
    if name == "ml-100k":
        extract_path = os.path.join(base_path, "ml-100k-sy")
        ratings = pd.read_csv(
            os.path.join(extract_path, "u.data"),
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"]
        )
        movies = pd.read_csv(
            os.path.join(extract_path, "u.item"),
            sep="|", header=None, encoding="latin-1"
        )
        movies = movies[[0, 1]]
        movies.columns = ["item_id", "title"]
        df = ratings.merge(movies, on="item_id")
    
    elif name == "ml-1m":
        ratings = pd.read_csv(
            os.path.join(base_path, "ml-1m/ratings.dat"),
            sep="::", header=None, engine="python",
            names=["user_id", "item_id", "rating", "timestamp"]
        )
        df = ratings
    
    elif name == "citeulike":
        # assumes citeulike.load_feedback() is available
        feedback = citeulike.load_feedback()
        df = pd.DataFrame(feedback, columns=["user_id", "item_id", "rating"])
    
    elif name == "dblp":
        train_path = os.path.join(base_path, "dblp/rating_train.dat")
        test_path  = os.path.join(base_path, "dblp/rating_test.dat")
        train_df = pd.read_csv(train_path, sep="\t", header=None, names=["user_id", "item_id", "rating"])
        test_df  = pd.read_csv(test_path, sep="\t", header=None, names=["user_id", "item_id", "rating"])
        df = pd.concat([train_df, test_df], ignore_index=True)
    
    elif name == "wiki":
        ratings_file = os.path.join(base_path, "wiki/ratings.dat")
        df = pd.read_csv(
            ratings_file,
            sep="\t",  # adjust to " " if required
            header=None, names=["user_id", "item_id", "rating"]
        )
    
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    
    return df