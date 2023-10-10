import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.distributions import constraints
from pyro.infer import SVI
from pyro.optim import Adam

from model.decoder import EuclideanDecoder
from model.distributions import GaussianRandomWalk

"""
Variational inference in the CLPM model With Pyro
"""


class TGNE:
    """
    Class for fitting a variational approximation in the CLPM model using Pyro
    Contains the forward model and guide, as well as a training Template
    """

    def __init__(
        self,
        n_nodes,
        dim=2,
        n_ticks=15,
        prior_scale=1.0,
        prior_scale_init=1.0,
        lr_bias=1e-5,
        lr_z=1e-2,
        neg_sampling=False,
        cuda=False,
    ):
        self.n_nodes = n_nodes
        self.dim = dim
        self.n_ticks = n_ticks
        self.ticks = torch.linspace(0, 1, n_ticks)
        self.z_init = torch.randn(n_nodes, dim, n_ticks)
        self.loc_init = torch.zeros(self.dim)
        self.prior_scale = prior_scale
        self.prior_scale_init = prior_scale_init

        self.loss = (
            pyro.infer.TraceMeanField_ELBO()
        )  # Uses the MCMC estimator only for the expected log likelihood term
        self.neg_sampling = neg_sampling

        # self.neg_sampling = negative_sampling
        pyro.clear_param_store()
        use_cuda = cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        print('Using device "{}"'.format(self.device))

        def get_adam_params(name):
            if "bias" in name:
                return {"lr": lr_bias}
            elif "loc" in name:
                return {"lr": 0.1}
            elif "scale" in "name":
                return {"lr": 0.01}
            else:
                return {"lr": lr_z}

        self.optimizer = Adam(get_adam_params)

        self.edge_index = None
        self.svi = SVI(self.model, self.guide, self.optimizer, loss=self.loss)

        self.params = nn.ParameterDict(
            {
                "bias": torch.tensor(
                    0.1,
                    requires_grad=True,
                    device=self.device,
                ),
                "z_loc": torch.zeros(
                    self.n_nodes,
                    self.dim,
                    self.n_ticks,
                    requires_grad=True,
                    device=self.device,
                ),
                "z_scale": torch.ones(
                    self.n_nodes,
                    self.dim,
                    self.n_ticks,
                    requires_grad=True,
                    device=self.device,
                ),
            }
        )
        self.param_namedict = {
            "bias": "AutoGuideList.0.bias",
            "z_loc": "AutoGuideList.1.loc",
            "z_scale": "AutoGuideList.1.scale",
        }

    def model(self, *batch):
        """
        Calculate the negative log-likelihood of the CLPM model
        """
        z_dist = GaussianRandomWalk(
            loc=self.loc_init,
            scale=self.prior_scale,
            scale_init=self.prior_scale_init,
            n_ticks=self.n_ticks,
        )
        bias = pyro.sample("bias", dist.Gamma(1.0, 1.0)).to(self.device)

        with pyro.plate("nodes", self.n_nodes):
            z = pyro.sample("z", z_dist).to(self.device)

        log_likelihood = EuclideanDecoder(z=z, bias=bias).log_likelihood(*batch)

        pyro.factor("clpm_log_likelihood", log_likelihood)

        assert not log_likelihood.isnan().any()
        assert log_likelihood.isfinite().all()
        loss = -log_likelihood  # negative log likelihood

        return loss  # negative log likelihood

    def guide(self, *batch):
        bias = pyro.param(
            "bias_map",
            self.params["bias"],
            constraint=constraints.positive,
        )
        pyro.sample("bias", dist.Delta(bias))
        # Normal Mean Field for the

        z_loc = pyro.param("z_loc", self.params["z_loc"])
        z_scale = pyro.param(
            "z_scale",
            self.params["z_scale"],
            constraint=constraints.positive,
        )
        z_dist = dist.Normal(z_loc, z_scale).to_event(2)
        with pyro.plate("nodes"):
            z = pyro.sample("z", z_dist)

        return z

    def training_step(self, batch):
        with pyro.poutine.trace(param_only=True) as param_capture:
            loss = self.svi.loss_and_grads(self.model, self.guide, *batch)
        param_dict = {
            name: site["value"].unconstrained()
            for name, site in param_capture.trace.nodes.items()
        }
        params = list(param_dict.values())
        grad_dict = {
            name: param.grad.norm().item() for name, param in param_dict.items()
        }

        self.svi.optim(list(param_dict.values()))

        logs = {
            "loss": loss,
            "grads": grad_dict,
        }

        pyro.infer.util.zero_grads(params)

        return logs

    @property
    def z_loc(self):
        # name = "AutoGuideList.1.loc"
        name = "z_loc"
        z_loc = pyro.param(name).view(self.n_nodes, self.dim, self.n_ticks)
        return z_loc

    @property
    def z_scale(self):
        # name = "AutoGuideList.1.scale"
        name = "z_scale"
        z_scale = pyro.param(name).view(self.n_nodes, self.dim, self.n_ticks)
        return z_scale

    @property
    def bias(self):
        # name = "AutoGuideList.0.bias"
        name = "bias_map"

        return pyro.param(name)

    @property
    def decoder(self):
        return EuclideanDecoder(self.z_loc, self.bias)

    def state_dict(self):
        return {
            "z_loc": self.z_loc,
            "z_scale": self.z_scale,
            "bias": self.bias,
        }

    def load(self, path):
        pyro.get_param_store().load(path)

        return self

    @torch.no_grad()
    def get_posterior_scores(self, src, dst, t, n_samples=100):
        z_dist = dist.Normal(self.z_loc, self.z_scale).to_event(2)

        z_B = z_dist.sample((n_samples,))
        y_score_B = torch.stack(
            [
                EuclideanDecoder(z_B[b], self.bias)(src, dst, t).exp()
                for b in range(z_B.shape[0])
            ]
        )

        return y_score_B.detach()  # Shape[batch, n_samples]


import numpy as np


def get_full_edge_index(num_nodes, directed=False):
    max_multi_idx = np.ravel_multi_index(
        (num_nodes - 1, num_nodes - 1),
        (num_nodes, num_nodes),
    )

    edge_multi_idx = np.arange(0, max_multi_idx)
    src, dst = np.unravel_index(
        edge_multi_idx,
        (num_nodes, num_nodes),
    )
    edge_index = np.vstack([src, dst])

    if directed:
        mask = edge_index[0] != edge_index[1]
    else:
        mask = edge_index[0] < edge_index[1]

    return (edge_index[:, mask], edge_multi_idx[mask])


def get_labeled_edge_index(src, dst, num_nodes, directed=False):
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

    return edge_index, edge_label


if __name__ == "__main__":
    import pandas as pd

    # Generate toy set of events
    events = pd.DataFrame(
        {
            "src": [0, 0, 0, 1, 1, 2, 2, 3, 3, 4],
            "dst": [1, 2, 3, 2, 3, 3, 4, 4, 5, 5],
            "t": [0.1, 0.34, 0.5, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
    )

    num_nodes = 6
    # Label the full set of edges with 0 and 1s. Used to calculate the partition function
    edge_index, edge_label = get_labeled_edge_index(
        events.src.values, events.dst.values, num_nodes
    )
    src, dst, t, edge_index, edge_label = (
        torch.tensor(events.src.values),
        torch.tensor(events.dst.values),
        torch.tensor(events.t.values),
        torch.tensor(edge_index),
        torch.tensor(edge_label),
    )
    tgne = TGNE(
        n_nodes=6,
        n_ticks=15,
        dim=2,
        prior_scale=1.0,
        prior_scale_init=1.0,
        lr_bias=1e-5,
        lr_z=1e-2,
    )

    negative_log_likelihood = tgne.model(src, dst, t, edge_index, edge_label)
    print(f"Negative log likelihood: {negative_log_likelihood}")
