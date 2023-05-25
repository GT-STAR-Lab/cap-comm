import json
import os
import pandas as pd
sacred_directories = [
    "/home/dwalkerhowell3/star_lab/ca-gnn-marl/results/sacred_runs/PredatorCapturePreyGNN-v0/1",
    "/home/dwalkerhowell3/star_lab/ca-gnn-marl/results/sacred_runs/PredatorCapturePreyGNN-v0/2",
    "/home/dwalkerhowell3/star_lab/ca-gnn-marl/results/sacred_runs/PredatorCapturePreyGNN-v0/3",
    "/home/dwalkerhowell3/star_lab/ca-gnn-marl/results/sacred_runs/PredatorCapturePreyGNN-v0/4",
    "/home/dwalkerhowell3/star_lab/ca-gnn-marl/results/sacred_runs/PredatorCapturePreyGNN-v0/5",
    "/home/dwalkerhowell3/star_lab/ca-gnn-marl/results/sacred_runs/PredatorCapturePreyGNN-v0/6",
    "/home/dwalkerhowell3/star_lab/ca-gnn-marl/results/sacred_runs/PredatorCapturePreyGNN-v0/7",
    "/home/dwalkerhowell3/star_lab/ca-gnn-marl/results/sacred_runs/PredatorCapturePreyGNN-v0/8",
    "/home/dwalkerhowell3/star_lab/ca-gnn-marl/results/sacred_runs/PredatorCapturePreyGNN-v0/9",
    "/home/dwalkerhowell3/star_lab/ca-gnn-marl/results/sacred_runs/PredatorCapturePreyGNN-v0/10",
    "/home/dwalkerhowell3/star_lab/ca-gnn-marl/results/sacred_runs/PredatorCapturePreyGNN-v0/11",
    "/home/dwalkerhowell3/star_lab/ca-gnn-marl/results/sacred_runs/PredatorCapturePreyGNN-v0/12",
    "/home/dwalkerhowell3/star_lab/ca-gnn-marl/results/sacred_runs/PredatorCapturePreyGNN-v0/13",
    "/home/dwalkerhowell3/star_lab/ca-gnn-marl/results/sacred_runs/PredatorCapturePreyGNN-v0/14",
    "/home/dwalkerhowell3/star_lab/ca-gnn-marl/results/sacred_runs/PredatorCapturePreyGNN-v0/15",
    "/home/dwalkerhowell3/star_lab/ca-gnn-marl/results/sacred_runs/PredatorCapturePreyGNN-v0/16",
]

hyperparameters_to_list = ["lr", "standardise_rewards", "target_update_interval_or_tau",
                           "entropy_coef"]

data = {hp: [] for hp in hyperparameters_to_list}
for dir in sacred_directories:
    with open(os.path.join(dir, "config.json"), 'r') as file:
        json_data = json.load(file)

    for hp in hyperparameters_to_list:
        data[hp].append(json_data[hp])

df = pd.DataFrame(data)
print(df)

df.to_csv("../results/grid_search_configs_list.csv")
