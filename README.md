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

See the file ``config.py`` for all configurations of the experiments. The 
scripts may take some time to run, to reduce computations decrease the number 
of iterations ``niter`` in the configurations.

Figures 1 to 5 can be reproduced with

```
python experiment_scaling_initialization.py
```

Figures 6 and 7 can be reproduced with

```
python experiment_regularity_and_scaling_initialization.py
```

Figure 9 can be reproduced by first training a grid of networks with different 
initializations and scalings on MNIST and CIFAR with

```
python main.py
```

Note that this script uses parallelization. Then, to plot the results,

```
python experiment_regularity_and_scaling_after_training.py
```
