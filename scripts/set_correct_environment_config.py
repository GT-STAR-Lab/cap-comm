import os
import sys
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = "../mpe/mpe/scenarios/configs"

def update_base_config(environment, agent_type):
    # Create directory if it doesn't exist
    directory_path = os.path.join(current_dir, base_dir, environment)
    os.makedirs(directory_path, exist_ok=True)

    # Path to base_config.yaml
    base_config_path = os.path.join(directory_path, 'base_config.yaml')

    # Load base_config.yaml
    with open(base_config_path, 'r') as file:
        config_data = yaml.safe_load(file)

    # Update 'override_base_config' parameter
    config_data['override_base_config'] = agent_type

    # Save updated config back to base_config.yaml
    with open(base_config_path, 'w') as file:
        yaml.dump(config_data, file, default_flow_style=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <environment> <agent_type>")
        sys.exit(1)
    
    environment = sys.argv[1]
    agent_type = sys.argv[2]

    update_base_config(environment, agent_type)
    print("base_config.yaml updated successfully.")
