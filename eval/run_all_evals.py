import subprocess
import os
import time

eval_dict = {
    "eval_3_agents_unseen_bc_default.yaml" : {
        "SC_4_agents_HSN" : [1, 2, 3],
        "MLP_4_agents_HSN" : [1, 2, 3],
        "SC_CASKIP_4_agents_HSN" : [1, 2, 3]
    },
    "eval_4_agents_unseen_bc_default.yaml" : {
        "SC_4_agents_HSN" : [1, 2, 3],
        "MLP_4_agents_HSN" : [1, 2, 3],
        "SC_CASKIP_4_agents_HSN" : [1, 2, 3]
    },
    "eval_4_agents_seen_bc_default.yaml" : {
        "SC_4_agents_HSN" : [1, 2, 3],
        "MLP_4_agents_HSN" : [1, 2, 3],
        "SC_CASKIP_4_agents_HSN" : [1, 2, 3]
    },
    "eval_5_agents_unseen_bc_default.yaml" : {
        "SC_4_agents_HSN" : [1, 2, 3],
        "MLP_4_agents_HSN" : [1, 2, 3],
        "SC_CASKIP_4_agents_HSN" : [1, 2, 3]
    },
    "eval_4_agents_CA_unseen_bc_default.yaml" : {
        "SC_4_agents_HSN" : [1, 2, 3],
        "MLP_4_agents_HSN" : [1, 2, 3],
        "SC_CASKIP_4_agents_HSN" : [1, 2, 3]
    },
    "eval_4_agents_ID_unseen_bc_default.yaml" : {
        "SC_ID_4_agents_REDO" : [1, 2, 3],
        "MLP_ID_4_agents_HSN" : [4, 5, 6]
    },
    "eval_4_agents_ID_seen_bc_default.yaml" : {
        "SC_ID_4_agents_REDO" : [1, 2, 3],
        "MLP_ID_4_agents_HSN" : [4, 5, 6]
    },
    "eval_3_agents_CA_unseen_bc_default.yaml" : {
        "SC_4_agents_HSN" : [1, 2, 3],
        "MLP_4_agents_HSN" : [1, 2, 3],
        "SC_CASKIP_4_agents_HSN" : [1, 2, 3]
    },
    "eval_5_agents_CA_unseen_bc_default.yaml" : {
        "SC_4_agents_HSN" : [1, 2, 3],
        "MLP_4_agents_HSN" : [1, 2, 3],
        "SC_CASKIP_4_agents_HSN" : [1, 2, 3]
    },
    "eval_3_agents_ID_unseen_bc_default.yaml" : {
        "SC_ID_4_agents_REDO" : [1, 2, 3],
        "MLP_ID_4_agents_HSN" : [4, 5, 6]
    },
    "eval_5_agents_ID_unseen_bc_default.yaml" : {
        "SC_ID_4_agents_REDO" : [1, 2, 3],
        "MLP_ID_4_agents_HSN" : [4, 5, 6]
    },
       
}

if __name__ == "__main__":
    
    python_eval_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval.py")

    subprocesses = [] # list to hold the subprocesses
    max_subprocesses = 16
    current_number_of_suprocesses = 0
    total_ran = 0
    # begin evaluations
    for eval_config_filename, models_dict in eval_dict.items():
        for model_name, seed_list in models_dict.items():
            for seed in seed_list:
                total_ran += 1

                args = ["--env_config_filename", eval_config_filename, "--sacred_run", str(seed),"--experiment_name", model_name]
                cmd = ['python', python_eval_path] + args
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