import subprocess
import os

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
    },
    "eval_4_agents_ID_seen_bc_default.yaml" : {
        "SC_ID_4_agents_REDO" : [1, 2, 3],
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
    },
    "eval_5_agents_ID_unseen_bc_default.yaml" : {
        "SC_ID_4_agents_REDO" : [1, 2, 3],
    },
       
}

if __name__ == "__main__":
    
    python_eval_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval.py")

    subprocesses = [] # list to hold the subprocesses
    max_subprocesses = 8
    current_number_of_suprocesses = 0

    # begin evaluations
    for eval_config_filename, models_dict in eval_dict.items():
        for model_name, seed_list in models_dict.items():
            for seed in seed_list:

                args = ["--env_config_filename", eval_config_filename, "--sacred_run", str(seed),"--experiment_name", model_name]
                cmd = ['python', python_eval_path] + args

                if(current_number_of_suprocesses < max_subprocesses):
                    proc = subprocess.Popen(cmd)
                    subprocesses.append(proc)
                    current_number_of_suprocesses += 1

                # wait for all subprocesses to complete
                else:
                    for proc in subprocesses:
                        proc.wait()

                    # reset process count
                    current_number_of_suprocesses = 0
    
    # wait for all subprocesses to complete
    for proc in subprocesses:
        proc.wait()

    print("All subprocesses have finished.")