from gym.envs.registration import register
import mpe.scenarios as scenarios
import time
import os
import yaml
# Multiagent envs
# ----------------------------------------

class DictView(object):
    def __init__(self, d):
        self.__dict__ = d
    def __str__(self):    
        return(str(self.__dict__))
        
_particles = {
    "multi_speaker_listener": "MultiSpeakerListener-v0",
    "simple_adversary": "SimpleAdversary-v0",
    "simple_crypto": "SimpleCrypto-v0",
    "simple_push": "SimplePush-v0",
    "simple_reference": "SimpleReference-v0",
    "simple_speaker_listener": "SimpleSpeakerListener-v0",
    "simple_spread": "SimpleSpread-v0",
    "simple_tag": "SimpleTag-v0",
    "simple_world_comm": "SimpleWorldComm-v0",
    "climbing_spread": "ClimbingSpread-v0",
    "terrain_aware_navigation": "TerrainAwareNavigation-v0",
    "heterogeneous_sensor_network": "HeterogeneousSensorNetwork-v0",
    "heterogeneous_material_transport": "HeterogeneousMaterialTransport-v0",
    "terrain_aware_navigation_ca": "TerrainAwareNavigationCA-v0",
    # "heterogeneous_sensor_network_ca": "HeterogeneousSensorNetworkCA-v0",
    # "heterogeneous_material_transport_ca": "HeterogeneousMaterialTransportCA-v0",
    "terrain_dependant_navigation": "TerrainDependantNavigation-v0",
    "search_and_capture": "SearchAndCapture-v0",
    "material_transport": "MaterialTransport-v0"
}
_environmnets_with_configs = [
     "material_transport"
]

current_dir = os.path.dirname(os.path.abspath(__file__))

for scenario_name, gymkey in _particles.items():

    
    # load a config for environments that have them.
    if(scenario_name in _environmnets_with_configs):
        
        config_dir = os.path.join(current_dir, 'scenarios', 'configs', scenario_name)
        with open(os.path.join(config_dir, 'base_config.yaml'), 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        # When an override config is provided within config, it will override parameters 
        # in the base config with the parameter in the override config.
        if config["override_base_config"]:
            with open(os.path.join(config_dir, config["override_base_config"]), 'r') as f:
                 override_config = yaml.load(f, Loader=yaml.SafeLoader)
            for key, value in override_config.items():
                if key in config:
                    config[key] = value
        config = DictView(config)
        scenario = scenarios.load(scenario_name + ".py").Scenario(config=config)

    else:
         scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()

    kwargs={
            "world": world,
            "reset_callback": scenario.reset_world,
            "reward_callback": scenario.reward,
            "observation_callback": scenario.observation
    }

    if(scenario_name == "material_transport"):
         kwargs["done_callback"] = scenario.done
         kwargs["info_callback"] = scenario.info
    # Registers multi-agent particle environments:
    register(
        gymkey,
        entry_point="mpe.environment:MultiAgentEnv",
        kwargs=kwargs
    )

# Registers the custom double spread environment:

for N in range(2, 11, 2):
    scenario_name = "simple_doublespread"
    gymkey = f"DoubleSpread-{N}ag-v0"
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world(N)

    register(
        gymkey,
        entry_point="mpe.environment:MultiAgentEnv",
        kwargs={
            "world": world,
            "reset_callback": scenario.reset_world,
            "reward_callback": scenario.reward,
            "observation_callback": scenario.observation,
        },
    )
