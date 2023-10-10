import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import torch
from pyro.distributions.torch_distribution import TorchDistribution


class GaussianRandomWalk(TorchDistribution):
    arg_constraints = {"loc_init": constraints.real_vector, "scale": constraints.real}

    def __init__(
        self,
        loc=torch.zeros(2),
        n_ticks=15,
        scale=0.1,
        scale_init=1.0,
    ):
        self.n_ticks = n_ticks * loc.new_ones(1).int()
        self.scale_init = scale_init
        self.scale = loc.new_ones(1) * scale / torch.sqrt(self.n_ticks)
        self.loc = loc
        self.obs_dim = self.loc.shape

        super().__init__(
            validate_args=False,
            event_shape=self.loc.shape + (self.n_ticks,),
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.real

    def sample(self, sample_shape=torch.Size()):
        """The right-most dimension is used as a time dimension"""
        # The last dimension should be the time dimension
        noise_loc = torch.zeros_like(self.loc)[..., None]
        noise_dist = dist.Normal(noise_loc, 1).expand(self.event_shape)
        noise = noise_dist.sample(sample_shape)
        terms = noise
        terms[..., 0] *= self.scale_init
        terms[..., 0] += self.loc
        terms[..., 1:] *= self.scale
        out = terms.cumsum(dim=-1)

        return out

    def log_prob(self, x):
        locs = torch.zeros_like(x)
        locs[..., 0] = self.loc
        scale = torch.ones_like(x)
        scale[..., 0] = self.scale_init
        scale[..., 1:] = self.scale
        # * self.scale
        locs[..., 1:] = x[..., :-1]
        # nd = dist.Normal(locs, self.scale).to_event(self.event_dim)
        nd = dist.Normal(locs, scale).to_event(self.event_dim)
        return nd.log_prob(x)


if __name__ == "__main__":
    with pyro.plate("node", 10):
        grw = GaussianRandomWalk(loc=torch.zeros(3, 2), scale=1.0, n_ticks=15)
        mus = pyro.sample("mu", grw)

    print(grw.log_prob(mus).shape)
    