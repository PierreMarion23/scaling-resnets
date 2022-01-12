# Training very deep resnets

## Installing dependencies

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install click numpy scipy matplotlib wandb pytorch-lightning -c conda-forge
conda install plotly -c plotly
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