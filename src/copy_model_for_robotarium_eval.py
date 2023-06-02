import os
import shutil
import argparse
import datetime
import json

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=str)
    parser.add_argument("--run-index", type=int)
    parser.add_argument("--environment", type=str)
    args = parser.parse_args()

    args.run_index = str(args.run_index)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    robotarium_scenario_dir = os.path.join(current_dir, "..", "Heterogeneous-MARL-CA", "robotarium_gym", "scenarios", args.environment.split("-v0")[0])

    sacred_run_dir = os.path.join(args.experiment_dir, "results", "sacred_runs", args.environment, args.run_index)

    run_config_file = os.path.join(sacred_run_dir, 'config.json')

    # open config
    with open(run_config_file, 'r') as file:
        config_data = json.load(file)

    unique_token = config_data['unique_token']
    env = config_data["env_args"]["key"]
    # get the model (last model)
    models_path = os.path.join(args.experiment_dir, "results", "models", env, unique_token, args.run_index)
    directories = [name for name in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, name))]
    sorted_directories = sorted(directories, key=lambda x: int(x))
    final_model_subdir = sorted_directories[-1]
    model_weights = os.path.join(models_path, final_model_subdir, "agent.th")

    # get the agent file
    agent_module_dir = os.path.join(args.experiment_dir, "modules", "agents")
    if config_data["agent"] == "gnn":
        agent_module = os.path.join(agent_module_dir, "gnn_agent.py")
    elif config_data["agent"] == "dual_channel_gnn":
        agent_module = os.path.join(agent_module_dir, "gnn_agent.py")

    # package the weights, agent file, and config into a single directory called model
    model_dir = os.path.join(robotarium_scenario_dir, "models")

    # Check if "model" directory exists, create it if it doesn't
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Copy files into "model" directory
    shutil.copy2(run_config_file, model_dir)
    shutil.copy2(model_weights, model_dir)
    # shutil.copy2(agent_module, model_dir)

    # copy config.yaml to model directory
    robotarium_env_config = os.path.join(args.experiment_dir, "Heterogeneous-MARL-CA", 
                                         "robotarium_gym", "scenarios", args.environment.split("-v0")[0], "config.yaml")


    shutil.copy2(robotarium_env_config, model_dir)

    

