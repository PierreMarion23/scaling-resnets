# Scaling ResNets in the Large-depth Regime

## Environment

### With conda

```
conda env create -f environment.yml
```

### With pip

Install Python 3.9.9 and pip 21.3.1, then

```
pip3 install -r requirements.txt
```

## Reproducing the paper figures

See the file ``config.py`` for all configurations of the experiments. The 
scripts may take some time to run. To reduce computation time decrease the number 
of iterations ``niter`` or the number of epochs ``epoch`` in the configurations.

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

Figure 9 can be reproduced with

```
python experiment_regularity_and_scaling_after_training.py
```
