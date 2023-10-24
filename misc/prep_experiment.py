"""
This helper file copies all of the related files to 
an experiment into one folder to use for later use
"""
import os
import shutil
import argparse
import datetime



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--environment", type=str)
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(current_dir, "..", "..", "experiments_ca-gnn-marl", args.name)
    
    unique_token = f"{datetime.datetime.now().strftime('%Y-%m-%d:%I-%M-%S-%p')}"
    # output_dir += "_" + unique_token


    # copy the agent and critic modules
    item = os.path.join("modules", "agents")
    dest_dir = os.path.join(output_dir, item)
    modules_dir = os.path.join(current_dir, item)
    shutil.copytree(modules_dir, dest_dir, copy_function=shutil.copy)

    item = os.path.join("modules", "critics")
    dest_dir = os.path.join(output_dir, item)
    modules_dir = os.path.join(current_dir, item)
    shutil.copytree(modules_dir, dest_dir, copy_function=shutil.copy)

    # copy the scenario config
    item = os.path.join("Heterogeneous-MARL-CA", "robotarium_gym", "scenarios", args.environment, "config.yaml")
    dest_dir = os.path.join(output_dir, item)
    scenario_dir = os.path.join(current_dir, "..", item)
    # shutil.copytree(scenario_dir, dest_dir, copy_function=shutil.copy)
    shutil.copy(item, output_dir)

    # create a blank text file
    note_file = "experiment_notes.txt"
    with open(os.path.join(output_dir, note_file), 'w') as file:
        pass