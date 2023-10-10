import numpy as np
import torch
import torch.utils.data as data


def get_labeled_edge_index(src, dst, num_nodes, directed=True):
    """
    src: torch.tensor

    dst: torch.tensor

    num_nodes: int
    """
    pos_idx = np.ravel_multi_index(
        (src, dst),
        (num_nodes, num_nodes),
    )
    max_multi_idx = np.ravel_multi_index(
        (num_nodes - 1, num_nodes - 1),
        (num_nodes, num_nodes),
    )
    full_multi_idx = np.arange(max_multi_idx + 1)
    full_labels = np.in1d(full_multi_idx, pos_idx)
    full_src, full_dst = np.unravel_index(
        full_multi_idx,
        (num_nodes, num_nodes),
    )

    if directed:
        mask = full_src != full_dst
    else:
        mask = full_src < full_dst
    edge_index = np.vstack([full_src[mask], full_dst[mask]])
    edge_label = full_labels[mask]

    return torch.tensor(edge_index), torch.tensor(edge_label)


def get_unique_edges(src, dst):
    """
    src: torch.tensor

    dst: torch.tensor
    """
    edge_index = torch.vstack([src, dst])
    return torch.unique(edge_index, dim=1)


class TemporalData(data.Dataset):
    """
    A class for managing Temporal Network Datasets
    Implements indexing by node, and collate for batching by node
    """

    def __init__(
        self,
        src,
        dst,
        t,
        num_nodes,
        precompute_edge_index=True,  # In the case where the number of nodes is small, we can precompute labels for each possible edge. Otherwise, we use negative sampling
    ):
        super().__init__()
        self.src = src
        self.dst = dst
        self.t = t
        self.num_nodes = num_nodes
        # must be implemented in the subclass
        if precompute_edge_index:
            self.edge_index, self.edge_label = get_labeled_edge_index(
                src.cpu().numpy(),
                dst.cpu().numpy(),
                num_nodes,
                directed=True,
            )
        else:
            self.edge_index, self.edge_label = None, None

    def __getitem__(self, idx):
        mask = np.in1d(self.src, idx)
        return (
            # Events src,dst,t
            self.src[mask],
            self.dst[mask],
            self.t[mask],
        )

    def __len__(self):
        return self.num_nodes

    def collate(self, batches):
        src = torch.hstack([b[0] for b in batches])
        dst = torch.hstack([b[1] for b in batches])
        t = torch.hstack([b[2] for b in batches])

        # Indices of the possible edges, along with labels
        # Indicating whether they are present or not
        mask = torch.isin(self.edge_index[0], src)
        edge_index = self.edge_index[:, mask]
        edge_label = self.edge_label[mask]

        return src.long(), dst.long(), t.float(), edge_index.long(), edge_label.long()


class TemporalDataNegativeSampling(TemporalData):
    """Class for Temporal Network data when the number of nodes is large enough than
    num_nodes **2 is larger than 1e6. In this case, we use negative sampling to
    reduce the number of labeled edges in the dataset.
    """

    def __init__(self, src, dst, t, num_nodes, num_neg_samples=1):
        super().__init__(src, dst, t, num_nodes, precompute_edge_index=False)
        self.num_neg_samples = num_neg_samples

    @staticmethod
    def neg_sampling(src, dst):
        """
        src: torch.tensor

        dst: torch.tensor
        """
        mask = torch.ones_like(src).bool()
        dst_neg = torch.empty_like(dst)
        for _ in range(10):
            dst_neg[mask] = torch.randint(
                0,
                num_nodes,
                dst[mask].shape,
                dtype=torch.long,
            )
            mask = torch.isin(dst_neg, dst)
            mask = dst_neg == src

            if mask.all():
                break

        return src, dst_neg

    def __getitem__(self, idx):
        mask = np.in1d(self.src, idx)
        return (
            # Events src,dst,t
            self.src[mask],
            self.dst[mask],
            self.t[mask],
        )

    def collate(self, batches):
        src = torch.hstack([b[0] for b in batches])
        dst = torch.hstack([b[1] for b in batches])
        t = torch.hstack([b[2] for b in batches])
        # Get the labeled edges
        ## First get the unique positive edges involved in the batch
        src_pos, dst_pos = get_unique_edges(src, dst)
        # For each positive edge, sample num_neg_samples negative edges
        src_neg, dst_neg = self.neg_sampling(src_pos, dst_pos)
        # Concatenate the positive and negative edges
        edge_index = torch.vstack(
            [
                torch.cat([src_pos, src_neg]),
                torch.cat([dst_pos, dst_neg]),
            ]
        )  # Shape [2, n_edges]
        # Label the positive edges as 1 and negative edges as 0
        edge_label = torch.hstack(
            [
                torch.ones_like(src_pos),
                torch.zeros_like(src_neg),
            ],
        )  # Shape [n_edges]
        return src.long(), dst.long(), t.float(), edge_index.long(), edge_label.long()


if __name__ == "__main__":
    # Some Tests
    import pandas as pd

    events = pd.DataFrame(
        {
            "src": [0, 0, 0, 1, 1, 2, 2, 3, 3, 4],
            "dst": [1, 2, 3, 2, 3, 3, 4, 4, 5, 5],
            "t": [0.1, 0.2, 0.3, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5],
        }
    )
    events.loc[:, ["src", "dst"]] = np.sort(
        events.loc[:, ["src", "dst"]].values, axis=1
    )
    src = torch.LongTensor(events["src"].values)
    dst = torch.LongTensor(events["dst"].values)
    t = torch.FloatTensor(events["t"].values)
    num_nodes = 6
    dataset = TemporalData(src, dst, t, num_nodes)

    # Look at the batches

    print(f"Data for Source Node 0:")
    for d in dataset[0]:
        print(d)

    # Look at the output of the collate function

    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=3,
        collate_fn=dataset.collate,
    )
    batch = next(iter(dataloader))
    print(f"Batch:")
    for b in batch:
        print(b)
    for src_, dst_, t_, edge_index_, edge_label_ in dataloader:
        assert (src_ < dst_).all()

    # Try out the Negative Sampling Class
    dataset = TemporalDataNegativeSampling(src, dst, t, num_nodes)

    print(f"Data for Source Node 0:")
    for d in dataset[0]:
        print(d)

    # Look at the output of the collate function

    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=3,
        collate_fn=dataset.collate,
    )
    batch = next(iter(dataloader))
    print(f"Batch:")
    for b in batch:
        print(b)
