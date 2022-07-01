# Scaling ResNets in the Large-depth Regime

## Installing dependencies

With pip:
```
pip3 install torch torchvision numpy scipy matplotlib multiprocessing pytorch-lightning
```

## Reproducing the paper figures

See the file ``config.py`` for all configurations of the experiments. The 
scripts may take some time to run. To reduce computation time decrease the number 
of iterations ``niter`` or the number of epochs ``epochin the configurations.

Figures 1 to 5 can be reproduced with

```
python experiment_scaling_initialization.py
```

Figures 6 and 7 can be reproduced with

```
python experiment_regularity_and_scaling_initialization.py
```
Figure 8 can be reproduced with

```
python experiment_weights_after_training.py
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
