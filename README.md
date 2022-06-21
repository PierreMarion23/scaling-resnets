# Scaling ResNets in the Large-depth Regime

## Installing dependencies

With conda:
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install click numpy scipy matplotlib wandb pytorch-lightning -c conda-forge
conda install plotly -c plotly
```

With pip:
```
pip3 install torch torchvision torchaudio click numpy scipy matplotlib wandb pytorch-lightning plotly
```

## Running the program

Debug:

```
python main.py -c debug -o
```

Standard:

```
python main.py
```

## Reproducing the paper figures

See the file ``config.py`` for all configurations of the experiments. 

Figures 1, 2 and 3 can be reproduced with

```
python scaling_init_experiment.py
```