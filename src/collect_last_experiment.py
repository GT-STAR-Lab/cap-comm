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
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(current_dir, "..", "..", "experiment_copies", args.name)
    
    unique_token = f"{datetime.datetime.now().strftime('%Y-%m-%d:%I-%M-%S-%p')}"
    output_dir += "_" + unique_token

    # copy all the config files
    item = "config"
    dest_dir = os.path.join(output_dir, item)
    config_dir = os.path.join(current_dir, item)
    shutil.copytree(config_dir, dest_dir, copy_function=shutil.copy)

    item = "scripts"
    dest_dir = os.path.join(output_dir, item)
    scripts_dir = os.path.join(current_dir, "..", item)
    shutil.copytree(scripts_dir, dest_dir, copy_function=shutil.copy)

    # copy the agent and critic modules
    item = os.path.join("modules", "agents")
    dest_dir = os.path.join(output_dir, item)
    modules_dir = os.path.join(current_dir, item)
    shutil.copytree(modules_dir, dest_dir, copy_function=shutil.copy)

    item = os.path.join("modules", "critics")
    dest_dir = os.path.join(output_dir, item)
    modules_dir = os.path.join(current_dir, item)
    shutil.copytree(modules_dir, dest_dir, copy_function=shutil.copy)

    # copy the results directory
    item = "results"
    dest_dir = os.path.join(output_dir, item)
    results_dir = os.path.join(current_dir, "..", item)
    shutil.copytree(results_dir, dest_dir, copy_function=shutil.copy)

    out = input("Would you like to delete the results current in the followig directory [y/N]:\n %s " % (results_dir))
    if(out == "y"):
        shutil.rmtree(results_dir)

    # copy the scenario directory
    item = os.path.join("Heterogeneous-MARL-CA", "robotarium_gym", "scenarios")
    dest_dir = os.path.join(output_dir, item)
    scenario_dir = os.path.join(current_dir, "..", item)
    shutil.copytree(scenario_dir, dest_dir, copy_function=shutil.copy)

    # create a blank text file
    note_file = "experiment_notes.txt"
    with open(os.path.join(output_dir, note_file), 'w') as file:
        pass