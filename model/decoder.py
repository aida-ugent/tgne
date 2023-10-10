import math
from abc import abstractmethod
from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributions import Normal

"""
This implements the CLPM model with Euclidean distance, viewed as a **decoder**
The decoder takes as input the embeddings of the source and destination nodes and calculates:
- The log rate
- The cumulative event rate: either using a closed form or using a Riemann approximation
- The log-likelihood of the event (difference between the two above)  

"""


class CLPMDecoder(nn.Module):
    """
    Generic class for CLPM decoder
    Subclasses can be defined in order to implement decoders with different similarity function (e.g. dot product, hyperbolic distance, etc...)
    """

    def __init__(self) -> None:
        super().__init__()
        self.z = None
        self.bias = None

    @abstractmethod
    def decode(self, z_src, z_dst, dim):
        pass

    @property
    def n_nodes(self):
        return self.z.shape[0]

    def forward(self, src, dst, t):
        """
        Return log the rate function of src, dst at time t
        Supports time-broadcasting:
        If src and dst have shape [batch,1] and t has shape [1,T] then the output will have shape [batch,T]
        """
        z = self.z
        n_ticks = z.shape[-1]
        step_len = 1 / (n_ticks - 1)
        time_index = torch.div(t, step_len, rounding_mode="floor").long()
        time_index = torch.clip(time_index, max=n_ticks - 2)
        delta_t = torch.remainder(t, step_len).float() / step_len
        one_m_delta_t = 1 - delta_t
        z_src_cur = z[src, ..., time_index]
        z_src_next = z[src, ..., time_index + 1]
        z_dst_cur = z[dst, ..., time_index]
        z_dst_next = z[dst, ..., time_index + 1]
        z_src = one_m_delta_t[..., None] * z_src_cur + delta_t[..., None] * z_src_next
        z_dst = one_m_delta_t[..., None] * z_dst_cur + delta_t[..., None] * z_dst_next
        logits = self.decode(z_src, z_dst)
        return logits

    def log_likelihood_full(self, src, dst, t, z):
        """Poisson Process log-likelihood, calculated as
        $ \sum_{\tau\in\Tcal_{ij}} \log(\lambda_{ij}(s)) -\int_{0}^{1} \lambda_{ij}(s) ds$
        src,dst,t stores the events
        """
        log_rates_events = self.forward(src, dst, t, z)
        pos_edge_index = torch.unique(torch.stack([src, dst], dim=0), dim=1)

        unique_src = src.unique()
        dst_all = torch.arange(self.n_nodes)
        edge_index = torch.cartesian_prod(unique_src, dst_all)
        mask = edge_index[:, 0] != edge_index[:, 1]
        edge_index = edge_index[mask]  # All possible edges
        src_e, dst_e = edge_index.T
        # Compute cumulative rate using the Riemann approximation
        cum_rate = self.cumul_rates_riemann(src_e, dst_e, z)

        ll = log_rates_events.sum()
        ll = ll - cum_rate.sum()
        return ll

    #
    def cumulative_rates_riemann(self, src, dst, t_start, t_end, M=100):
        """
        Calculate the cumulative rates of the edges (src,dst) between t_start and t_end
        using the Riemann approximation with M steps
        """
        lam = torch.linspace(0, 1, M + 1)[None, :]
        t_grid = t_start[:, None] + lam * (t_end - t_start)[:, None]
        rates = self.forward(src[:, None], dst[:, None], t_grid).exp()
        cum_rates = (t_grid.diff(dim=1) * rates[:, :-1]).sum(dim=1)
        return cum_rates  # Shape [n_edges]


RV = Normal(0, 1)


class EuclideanDecoder(CLPMDecoder):
    """Decoder for TGNE with Euclidean distance
    Contains the code for the conditional model of the interactions given the embeddings
    """

    def __init__(self, z, bias):
        super().__init__()
        self.z = z
        self.bias = bias
        # self.n_nodes = self.z.shape[0]
        self.n_combinations = self.n_nodes * (self.n_nodes - 1) // 2

    def decode(self, z_src, z_dst, dim=-1):
        z_dist = (z_src - z_dst).pow(2).sum(dim=dim)
        logits = self.bias - z_dist
        return logits

    def cumul_rates(self, edge_index):
        z = self.z
        n_ticks = self.z.shape[-1]
        src, dst = edge_index

        n_intervals = n_ticks - 1
        z_src = z[src]
        z_dst = z[dst]
        delta_z = z_src - z_dst

        delta_cur = delta_z[..., 1:]
        delta_next = delta_z[..., :-1]

        cur_dot_cur_min_next = (delta_cur * (delta_cur - delta_next)).sum(-2)
        cur_min_next_norm2 = (delta_cur - delta_next).pow(2).sum(-2)

        inv_cur_min_next_norm2 = 1 / (cur_min_next_norm2 + 1e-9)

        sigma = inv_cur_min_next_norm2.sqrt() / math.sqrt(2)

        mu = cur_dot_cur_min_next * inv_cur_min_next_norm2

        a = delta_cur.pow(2).sum(-2) - mu * cur_dot_cur_min_next

        step_size = 1 / n_intervals  # 1/number of intervals
        integral = (self.bias - a).exp()
        integral = integral * step_size
        integral = integral * sigma

        integral = integral * (RV.cdf((1 - mu) / sigma) - RV.cdf(-mu / sigma))
        return integral * math.sqrt(2 * math.pi)

    def log_likelihood(self, src, dst, t, edge_index, edge_label=None):
        """
        src,dst,t: events
        edge_index: edges on which to calculate the partition function
        """
        log_rates_events = self(src, dst, t)
        cum_rate = self.cumul_rates(edge_index).sum(dim=-1)
        if edge_label is not None:
            scaling = self.get_scaling_factors(edge_index, edge_label)
            cum_rate = cum_rate * scaling
        ll = log_rates_events.sum()
        ll = ll - cum_rate.sum()
        return ll

    def log_likelihood_riemann(
        self, src, dst, t, edge_index, edge_label=None, M_riemann=100
    ):
        """Poisson Process log-likelihood, calculated as
        $ \sum_{\tau\in\Tcal_{ij}} \log(\lambda_{ij}(s)) -\int_{0}^{1} \lambda_{ij}(s) ds$
        src,dst,t stores the events
        """

        logits = self.forward(src, dst, t)
        cum_rate = self.cumul_rates_riemann(edge_index[0], edge_index[1], M=M_riemann)
        # cum_rate = self.cumul_rates(edge_index)
        if edge_label is not None:
            mask = edge_label == 0
            cum_rate[mask] = cum_rate[mask] * self.n_combinations / mask.sum()

        ll = logits.sum()
        ll = ll - cum_rate.sum()
        return ll

    def get_scaling_factors(self, edge_index, edge_label):
        """
        For each node, calculate the ratio of the number of potential negative neighbors to the number of actual potential neighbors (as defined by edge label)
        """
        n_nodes = self.z.shape[0]

        num_pos = torch.zeros_like(self.z[:, 0, 0], dtype=torch.int64)
        nodes, freq = torch.unique(edge_index[0, edge_label == 1], return_counts=True)
        num_pos[nodes] = freq
        num_neg = n_nodes - 1 - num_pos
        num_neg + num_pos  # Number of potential neighbors

        # Calculate the actual number of negative neighbors
        num_neg_actual = torch.zeros(n_nodes, dtype=torch.int64)
        nodes, freq = torch.unique(edge_index[0, edge_label == 0], return_counts=True)
        num_neg_actual[nodes] = freq

        ratio = num_neg / num_neg_actual
        scale_factors = torch.ones_like(edge_label, dtype=torch.float32)
        scale_factors[edge_label == 0] = ratio[edge_index[0, edge_label == 0]]
        return scale_factors
