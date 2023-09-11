import subprocess
import os
import time
import sys

# eval_dict = {
#     # only non-id models can do unseen
#     # unseen 3 agents (CA)
#     "eval_3_agents_CA_unseen.yaml" : {
#         "GNN_CA_4_agents_MT" : [...],
#         "MLP_CA_4_agents_MT" : [...],
#         "GNN_CA_SKIP_4_agents_MT" : [...]
#     },
#     # unseen 4 agents (CA)
#     "eval_4_agents_CA_unseen.yaml" : {
#         "GNN_CA_4_agents_MT" : [...],
#         "MLP_CA_4_agents_MT" : [...],
#         "GNN_CA_SKIP_4_agents_MT" : [...]
#     },
#     # unseen 5 agents (CA)
#     "eval_5_agents_CA_unseen.yaml" : {
#         "GNN_CA_4_agents_MT" : [...],
#         "MLP_CA_4_agents_MT" : [...],
#         "GNN_CA_SKIP_4_agents_MT" : [...]
#     },
#     # seen 3 agents (CA)
#     "eval_3_agents_CA_seen.yaml" : {
#         "GNN_CA_4_agents_MT" : [...],
#         "MLP_CA_4_agents_MT" : [...],
#         "GNN_CA_SKIP_4_agents_MT" : [...]
#     },
#     # seen 3 agents (ID)
#     "eval_3_agents_ID_seen.yaml" : {
#         "GNN_ID_4_agents_MT" : [...],
#         "MLP_ID_4_agents_MT" : [...]
#     },
#     # seen 4 agents (CA)
#     "eval_4_agents_CA_seen.yaml" : {
#         "GNN_CA_4_agents_MT" : [...],
#         "MLP_CA_4_agents_MT" : [...],
#         "GNN_CA_SKIP_4_agents_MT" : [...]
#     },
#     # seen 4 agents (ID)
#     "eval_4_agents_ID_seen.yaml" : {
#         "GNN_ID_4_agents_MT" : [...],
#         "MLP_ID_4_agents_MT" : [...]
#     },
#     # seen 5 agents (CA)
#     "eval_5_agents_CA_seen.yaml" : {
#         "GNN_CA_4_agents_MT" : [...],
#         "MLP_CA_4_agents_MT" : [...],
#         "GNN_CA_SKIP_4_agents_MT" : [...]
#     },
#     # seen 5 agents (ID)
#     "eval_5_agents_ID_seen.yaml" : {
#         "GNN_ID_4_agents_MT" : [...],
#         "MLP_ID_4_agents_MT" : [...]
#     },     
# }

eval_dict = {
    # # only non-id models can do unseen
    # # ======================= UNSEEN AGENTS ==========================
    # # The following three evaluations are only for capability aware models
    # # since they can handle unseen agents.
    # # unseen 3 agents (CA)
    # "eval_3_agents_CA_unseen.yaml" : {
    #     "GNN_CA_4_agents_MT" : [1, 2, 43],
    #     "GNN_CA_SKIP_4_agents_MT" : [1, 2, 3],
    #     "MLP_CA_4_agents_MT" : [1, 2, 3]
    # },
    # # unseen 4 agents (CA)
    # "eval_4_agents_CA_unseen.yaml" : {
    #     "GNN_CA_4_agents_MT" : [1, 2, 43],
    #     "GNN_CA_SKIP_4_agents_MT" : [1, 2, 3],
    #     "MLP_CA_4_agents_MT" : [1, 2, 3]
    # },
    # # unseen 5 agents (CA)
    # "eval_5_agents_CA_unseen.yaml" : {
    #     "GNN_CA_4_agents_MT" : [1, 2, 43],
    #     "GNN_CA_SKIP_4_agents_MT" : [1, 2, 3],
    #     "MLP_CA_4_agents_MT" : [1, 2, 3]
    # },
    "eval_8_agents_CA_unseen.yaml" : {
        "GNN_CA_4_agents_MT" : [1, 2, 43],
        "GNN_CA_SKIP_4_agents_MT" : [1, 2, 3],
        "MLP_CA_4_agents_MT" : [1, 2, 3]
    },
    "eval_10_agents_CA_unseen.yaml" : {
        "GNN_CA_4_agents_MT" : [1, 2, 43],
        "GNN_CA_SKIP_4_agents_MT" : [1, 2, 3],
        "MLP_CA_4_agents_MT" : [1, 2, 3]
    },
    "eval_15_agents_CA_unseen.yaml" : {
        "GNN_CA_4_agents_MT" : [1, 2, 43],
        "GNN_CA_SKIP_4_agents_MT" : [1, 2, 3],
        "MLP_CA_4_agents_MT" : [1, 2, 3]
    },
    # #======================= UNSEEN TEAMS =============================
    # # The next evaluations are for teams of unseen teams, but seen agents
    # # thus agent IDs can participate
    # # unseen teams 3 agents (CA)
    # "eval_3_agents_CA_unseen_teams.yaml" : {
    #     "GNN_CA_4_agents_MT" : [1, 2, 43],
    #     "GNN_CA_SKIP_4_agents_MT" : [1, 2, 3],
    #     "MLP_CA_4_agents_MT" : [1, 2, 3]
    # },
    # # unseen teams 4 agents (CA)
    # "eval_4_agents_CA_unseen_teams.yaml" : {
    #     "GNN_CA_4_agents_MT" : [1, 2, 43],
    #     "GNN_CA_SKIP_4_agents_MT" : [1, 2, 3],
    #     "MLP_CA_4_agents_MT" : [1, 2, 3]
    # },
    # # unseen teams 5 agents (CA)
    # "eval_5_agents_CA_unseen_teams.yaml" : {
    #     "GNN_CA_4_agents_MT" : [1, 2, 43],
    #     "GNN_CA_SKIP_4_agents_MT" : [1, 2, 3],
    #     "MLP_CA_4_agents_MT" : [1, 2, 3]
    # },
    # # # unseen teams 3 agents (ID)
    # "eval_3_agents_ID_unseen_teams.yaml" : {
    #     "GNN_ID_4_agents_MT" : [1, 2, 54],
    #     "MLP_ID_4_agents_MT" : [1, 2, 3]
    # },
    # # unseen teams 4 agents (ID)
    # "eval_4_agents_ID_unseen_teams.yaml" : {
    #     "GNN_ID_4_agents_MT" : [1, 2, 54],
    #     "MLP_ID_4_agents_MT" : [1, 2, 3]
    # },
    # # unseen teams 5 agents (ID)
    # "eval_5_agents_ID_unseen_teams.yaml" : {
    #     "GNN_ID_4_agents_MT" : [1, 2, 54],
    #     "MLP_ID_4_agents_MT" : [1, 2, 3]
    # },
    
    # unseen teams 8 agents (CA)
    # "eval_8_agents_CA_unseen_teams.yaml" : {
    #     "GNN_CA_4_agents_MT" : [1, 2, 43],
    #     "GNN_CA_SKIP_4_agents_MT" : [1, 2, 3],
    #     "MLP_CA_4_agents_MT" : [1, 2, 3]
    # },

    # "eval_8_agents_ID_unseen_teams.yaml" : {
    #     "GNN_ID_4_agents_MT" : [1, 2, 54],
    #     "MLP_ID_4_agents_MT" : [1, 2, 3]
    # },

    # "eval_10_agents_CA_unseen_teams.yaml" : {
    #     "GNN_CA_4_agents_MT" : [1, 2, 43],
    #     "GNN_CA_SKIP_4_agents_MT" : [1, 2, 3],
    #     "MLP_CA_4_agents_MT" : [1, 2, 3]
    # },

    # "eval_10_agents_ID_unseen_teams.yaml" : {
    #     "GNN_ID_4_agents_MT" : [1, 2, 54],
    #     "MLP_ID_4_agents_MT" : [1, 2, 3]
    # },

    # "eval_15_agents_CA_unseen_teams.yaml" : {
    #     "GNN_CA_4_agents_MT" : [1, 2, 43],
    #     "GNN_CA_SKIP_4_agents_MT" : [1, 2, 3],
    #     "MLP_CA_4_agents_MT" : [1, 2, 3]
    # },

    # "eval_15_agents_ID_unseen_teams.yaml" : {
    #     "GNN_ID_4_agents_MT" : [1, 2, 54],
    #     "MLP_ID_4_agents_MT" : [1, 2, 3]
    # },
    #======================= SEEN TEAMS (TRAINING SET) ==========================

    # "eval_4_agents_CA_seen_teams.yaml" : {
    #     "GNN_CA_4_agents_MT" : [1, 2, 43],
    #     "GNN_CA_SKIP_4_agents_MT" : [1, 2, 3],
    #     "MLP_CA_4_agents_MT" : [1, 2, 3]
    # },
    # "eval_4_agents_ID_seen_teams.yaml" : {
    #     "GNN_ID_4_agents_MT" : [1, 2, 54],
    #     "MLP_ID_4_agents_MT" : [1, 2, 3]

    # },
}
num_episodes=1000
max_num_steps=100
render_freq=10
save_renders=False
render=False

og_args = ['--environment', "mpe:MaterialTransport-v0",
           '--num_episodes', str(num_episodes),
           '--max_num_steps', str(max_num_steps),
           '--render_freq', str(render_freq)]

if(save_renders):
    og_args += ['--save_renders']
if(render):
    og_args += ['--render']

if __name__ == "__main__":
    
    python_eval_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_mpe_material_transport.py")

    subprocesses = [] # list to hold the subprocesses
    max_subprocesses = 12
    current_number_of_suprocesses = 0
    total_ran = 0
    # begin evaluations
    for eval_config_filename, models_dict in eval_dict.items():
        for model_name, seed_list in models_dict.items():
            for seed in seed_list:
                total_ran += 1

                args = ["--eval_config_filename", eval_config_filename, "--sacred_run", str(seed),"--experiment_name", model_name]
                cmd = ['python', python_eval_path] + og_args + args
                print(cmd)
                proc = subprocess.Popen(cmd)
                subprocesses.append(proc)
                current_number_of_suprocesses += 1

                # wait for all subprocesses to complete
                if(current_number_of_suprocesses >= max_subprocesses):
                    for proc in subprocesses:
                        proc.wait()

                    # reset process count
                    subprocesses = []
                    current_number_of_suprocesses = 0
    
    # wait for all subprocesses to complete
    for proc in subprocesses:
        proc.wait()
    subprocesses = []
    print("All subprocesses have finished. Total: %d" % (total_ran))