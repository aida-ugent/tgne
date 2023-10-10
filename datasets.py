import os
from datetime import datetime

import numpy as np
import pandas as pd


def minmaxscale(x):
    return (x - min(x)) / (max(x) - min(x))


def load_highschool():
    events = pd.read_csv(
        "http://www.sociopatterns.org/wp-content/uploads/2014/08/highschool_2012.csv.gz",
        sep="\t",
        header=None,
        names=["timestamp", "src", "dst", "Ci", "Cj"],
    )

    nodes = pd.read_csv(
        "http://www.sociopatterns.org/wp-content/uploads/2015/09/metadata_2012.txt",
        sep="\t",
        header=None,
        names=["raw_id", "class", "gender"],
    ).reset_index(names=["id"])

    nodes.set_index("raw_id", inplace=True)
    events["src"] = nodes.loc[events["src"]].values
    events["dst"] = nodes.loc[events["dst"]].values
    # Make undirected
    events.loc[:, ["src", "dst"]] = np.sort(events[["src", "dst"]].values, axis=1)

    events.loc[:, "date"] = events["timestamp"].apply(
        lambda x: datetime.fromtimestamp(x).date()
    )

    # Make sure that the events are sorted by time
    events["t"] = events["timestamp"].values
    events.sort_values("t", inplace=True)
    events["t"] = minmaxscale(events["t"])
    events.reset_index(inplace=True, drop=True)

    return {"events": events, "nodes": nodes}


def load_toy():
    path = os.path.join("data", "simulated")
    # self.dataset = load_toy_dataset(device=device)
    fpath = os.path.join(path, "simul_B.npy")
    data_simul_B = np.load(fpath, allow_pickle=True)[()]

    nodes = data_simul_B["cluster_assignements"]  # (n_nodes, n_segments)
    events = pd.DataFrame(
        data_simul_B["edgelist"],
        columns=["t", "src", "dst"],
    )
    events = events[["src", "dst", "t"]].astype({"src": int, "dst": int, "t": float})

    n_segments = nodes.shape[1]
    seg_len = 1.0 / n_segments
    events["t"] = minmaxscale(events["t"])
    events["segment_idx"] = np.minimum(
        n_segments - 1,
        events.t.values // seg_len,
    ).astype(int)
    events.sort_values("t", inplace=True)

    events["src_cluster"] = nodes[events.src.values, events.segment_idx.values]
    events["dst_cluster"] = nodes[events.dst.values, events.segment_idx.values]

    return {"events": events, "nodes": nodes}


def load_dataset(name):
    if name == "toy":
        return load_toy()
    elif name == "highschool":
        return load_highschool()
    else:
        raise ValueError(f"Unknown dataset {name}")


def to_temporal_data(cls, events, num_nodes):
    import torch

    src = events["src"].values
    dst = events["dst"].values
    t = events["t"].values
    return cls(
        src=torch.tensor(src),
        dst=torch.tensor(dst),
        t=torch.tensor(t),
        num_nodes=num_nodes,
    )


def load_temporal_data(dataset):
    from temporal_data import TemporalData

    if dataset == "toy":
        from datasets import load_toy

        events, nodes = load_toy().values()

        return to_temporal_data(TemporalData, events, len(nodes))
    elif dataset == "highschool":
        from datasets import load_highschool

        events, nodes = load_highschool().values()
        events = events.set_index("date")
        first_date = events.index[0]
        events = events.loc[first_date]
        return to_temporal_data(TemporalData, events, len(nodes))
    else:
        raise ValueError(f"Unknown dataset {dataset}")


def get_adjbox(events, n_nodes, n_intervals):
    """
    Calculate a 3d array A of shape [n_nodes, n_nodes,n_intervals]
    such that for each triplet (i,j,k),
    A[i,j,k] = number of interactions between i and j in the interval k
    """
    A = np.zeros((n_nodes, n_nodes, n_intervals))
    df = events.copy()
    df["t"] = minmaxscale(df["t"])  # Make sure time is normalized between 0 and 1
    df["interval"] = pd.cut(df["t"], bins=n_intervals, labels=False).values
    df["interval"] = pd.cut(events["t"], n_intervals, labels=False)
    count_df = (
        df.groupby(["src", "dst", "interval"]).size().rename("count").reset_index()
    )
    A[count_df["src"], count_df["dst"], count_df["interval"]] = count_df["count"]
    return A


def get_interval_count_df(events, n_intervals):
    """
    Divide the time interval into n_interval bins
    For each bin, and each node pair, count the number of interactions
    for this node pair in the bin
    """


if __name__ == "__main__":
    print("Loading toy dataset")
    events, nodes = load_toy().values()
    print("toy events   : ", events)
    print("toy nodes    : ", nodes)
    print("Loading highschool dataset")
    events, nodes = load_highschool().values()

    print("highschool events   : ", events)
    print("highschool nodes    : ", nodes)
