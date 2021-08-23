from datetime import datetime
import json
import numpy as np
import torch

from scaling_experiments import run_experiment

DEFAULT_PARAMS = dict(
    delta_type="none",  # 'shared', 'multi', 'fixed'
    initial_sd=1.0e-04,
    init_method="xavier",
    activation="tanh",  # 'relu', 'tanh', 'linear'
    dim=10,
    dataset="ODE",  # 'ODE', 'mnist'
    optimizer_name="sgd",
    num_epochs=200,
    epsilon=1.0e-03,
    train_size=1024,
    test_size=256,
    batch_size=50,
    lr=1.0e-03,
    path="./scaling/",
    save=True,
    min_depth=3,
    max_depth=1000,
    base=1.2,  # base**n < max_depth
)

# SCALING_PARAMS = [
#     dict(
#         DEFAULT_PARAMS,
#         **dict(
#             path=DEFAULT_PARAMS["path"] + "dataset-mnist/act-tanh/delta-shared/",
#             dataset="ODE",
#             num_epochs=2000,
#             dim=10,
#             batch_size=1024,
#             lr=8.0,
#             epsilon=1e-5,
#             activation="tanh",
#             delta_type="fixed",  # 'multi', 'shared', 'fixed'
#             initial_sd=0.01,
#             init_method="zero",
#             min_depth=3,
#             max_depth=128,
#             base=2,
#             seed=42
#         ),
#     )
# ]


SCALING_PARAMS = [
    dict(
        DEFAULT_PARAMS,
        **dict(
            path=DEFAULT_PARAMS["path"] + "dataset-mnist-2/",
            dataset="mnist",
            num_epochs=500,
            dim=25,
            batch_size=60000,
            lr=8.0,
            epsilon=1e-5,
            activation="tanh",
            delta_type="fixed",  # 'multi', 'shared', 'fixed'
            initial_sd=0.01,
            init_method="zero",
            min_depth=3,
            max_depth=128,
            base=2,
            seed=42
        ),
    )
]


def run():
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"

    NOW = datetime.now().strftime("%Y-%m-%d-%H-%M") + "/"
    for i, params_dict in enumerate(SCALING_PARAMS):
        params_dict["path"] += NOW
        print(f"Scaling experiments, starting {i + 1}/{len(SCALING_PARAMS)}.")
        print("Path: ", params_dict["path"])
        params_dict["device"] = dev
        run_experiment(**params_dict)

        with open(params_dict["path"] + "params_dict.json", "w") as fp:
            json.dump(params_dict, fp)


if __name__ == "__main__":
    run()
