import random
from collections import defaultdict

import numpy as np
import torch
import torch_geometric
from sacred import Experiment
from sacred.observers import FileStorageObserver
from tqdm import trange

from model.tgne import TGNE
from temporal_data import TemporalData

ex = Experiment("tgne")
ex.observers.append(FileStorageObserver.create("my_runs"))


def set_seed(seed):
    print("Setting seed to", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)
    torch_geometric.seed_everything(seed)


def to_temporal_data(cls, events, num_nodes):
    src = events["src"].values
    dst = events["dst"].values
    t = events["t"].values
    return cls(
        src=torch.tensor(src),
        dst=torch.tensor(dst),
        t=torch.tensor(t),
        num_nodes=num_nodes,
    )


def load_dataset(dataset):
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


@ex.capture
@ex.automain
def train(
    dataset="toy",
    seed=42,
    dim=2,
    prior_scale_init=None,
    prior_scale=1.0,
    n_ticks=15,
    n_epochs=200,
    lr_z=0.1,
    lr_bias=0.001,
    cuda=False,
    batch_size=None,
    debug=False,
    save=True,
):
    temporal_data = load_dataset(dataset)

    if debug:
        print("Debug mode")
        n_epochs = 1
        save = False
    tgne = TGNE(
        n_nodes=temporal_data.num_nodes,
        n_ticks=n_ticks,
        dim=dim,
        prior_scale=prior_scale,
        prior_scale_init=prior_scale if prior_scale_init is None else prior_scale_init,
        cuda=cuda,
        lr_bias=lr_bias,
        lr_z=lr_z,
    )
    train_data = temporal_data
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=len(train_data) if not batch_size else batch_size,
        collate_fn=train_data.collate,
    )

    steps = trange(n_epochs)

    losses = []
    grads = defaultdict(list)
    for step in steps:
        for batch in train_loader:
            batch_logs = tgne.training_step(batch)
        loss = batch_logs["loss"]
        for param_name, grad in batch_logs["grads"].items():
            grads[param_name].append(grad)

        losses.append(batch_logs["loss"])
        steps.set_description("loss: {:.4f}".format(loss))
        ex.log_scalar("loss", loss, step)
        for param_name, grad in batch_logs["grads"].items():
            ex.log_scalar(f"grad_{param_name}", grad, step)

    out = {
        "state_dict": tgne.state_dict(),
        "losses": losses,
        "grads": grads,
    }

    if save:
        fname = f"tgne_{dataset}_{n_epochs}.ckpt"
        outpath = f"output/{fname}"
        torch.save(out, outpath)
        # This will copy and paste the trained model to the run-specific folder
        ex.add_artifact(outpath, name=fname)
