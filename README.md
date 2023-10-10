# Code for Temporal Gaussian Embedding of Temporal Networks

## Requirements

```
numpy==1.23.4
pandas==1.5.3
pyro==3.16
pyro_ppl==1.8.4
torch==1.12.1
tqdm==4.64.1
matplotlib==3.3.4
sacred==0.8.4
```

The `sacred` library is used in the training script to save the trained model in a run-specific folder, along with the config file and some metrics.
## Example training

The training can be run on different datasets by running the following commands

To train on the highschool dataset, run
```
python train.py with config/highschool.yaml
```

To train on the toy dataset, run

```
python train.py with config/toy.yaml
```

## Example analysis of the trained tgne models
In `notebooks`, we provide examples of how to visualize the 2d embeddings obtained through TGNE on the toy dataset and the highschool dataset. 


For any questions, please contact [Raphaël Romero](mailto:raphael.romero@ugent.be) at raphael.romero@ugent.be.
