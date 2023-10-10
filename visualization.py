import numpy as np
import pandas as pd
import seaborn as sns


def set_theme():
    sns.set_theme(
        context="notebook",
        font_scale=1.5,
        font="serif",
        style="darkgrid",
        rc={  #'figure.figsize':(6.4,4.8),
            "figure.dpi": 300,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "font.family": "serif",
            "axes.labelsize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 30,
            "axes.titlesize": 20,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
        },
    )


def get_pos_df(z_loc, z_scale):
    """
    Transforms the latent trajectory parameters into a dataframe
    containing the latent position for each node and tick

    Output:
    A dataframe with the following columns:
    - node: the node index
    - time_index: the index of the changepoint (between 0 and n_ticks)
    - value_x: the x value of the latent trajectory
    - value_y: the y value of the latent trajectory
    - relative_time:
    - size: the size of the node

    """
    n_nodes, dim, n_ticks = z_loc.shape

    z_2d = z_loc.transpose(0, 2, 1).reshape(-1, dim)
    ticks = np.linspace(0, 1, n_ticks)
    pos_df = pd.DataFrame(
        {
            "node": np.repeat(np.arange(n_nodes), n_ticks),
            "time_index": np.tile(np.arange(n_ticks), n_nodes),
            "value_x": z_2d[:, 0],
            "value_y": z_2d[:, 1],
            "tick_time": np.tile(ticks, n_nodes),
        }
    )

    sizes = np.linalg.norm(
        z_scale, axis=1
    )  # We use the norm of the Gaussian Scale as the node sizes
    pos_df["size"] = sizes[pos_df["node"], pos_df["time_index"]]
    return pos_df.sort_values(["node", "time_index"])
