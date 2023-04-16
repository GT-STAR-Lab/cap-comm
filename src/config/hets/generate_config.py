import yaml
import argparse
import numpy as np
import copy
def main(args):
    # Write YAML file
    # with open('data.yaml', 'w', encoding='utf8') as outfile:
    #     yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)

    # Read YAML file
    with open(args.default_yaml, 'r') as stream:
        config = yaml.safe_load(stream)

    config_keys = list(config.keys())
    num_candidates = config['num_train_candidates'] + config['num_test_candidates']
    idx_size = int(np.ceil(np.log2(num_candidates)))
    config['agents'] = {}
    for i in range(num_candidates):
        config['agents'][i] = {}
        config['agents'][i]['id'] = format(i, '#0'+str(idx_size + 2)+'b').replace('0b', '')
    
    #The way class probs works is the first index is the probability of class 1, 2nd is class 2, ...
    #The final index is the probability of all classes appearing
    if 'class_probs' in config_keys:
        agent_types = np.random.choice(range(len(config['class_probs'])), p=config['class_probs'], size=num_candidates)
    else:
        agent_types = []

    if 'traits' in config_keys:
        traits_keys = list(config['traits'])
        for t, trait in enumerate(traits_keys):
            # trait_mean = config['traits'][trait]['mean']
            # trait_var = config['traits'][trait]['var']

            func_args = copy.deepcopy(config['traits'][trait])
            del func_args['distribution']
            func_args['size'] = (num_candidates,)
            
            # Generate trait values by sampling from the distribution in the config
            trait_val = getattr(np.random, config['traits'][trait]['distribution'])(**func_args)

            if config['clipped']:
                trait_val[trait_val<=0] = 0
            idx_size = int(np.ceil(np.log2(num_candidates)))
            for i in range(num_candidates):
                if agent_types == []:
                    config['agents'][i][trait] = float(trait_val[i])
                else:
                    if agent_types[i] == len(config['class_probs'])-1 or agent_types[i] == t:
                        config['agents'][i][trait] = float(trait_val[i])
                    else:
                        config['agents'][i][trait] = 0

    with open(args.default_yaml if args.save_path is None else args.save_path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-dy', '--default-yaml', default='default_het_config.yaml')
    parser.add_argument('-d', '--dist', default=None)
    parser.add_argument('-p','--params', nargs='+', help='<Required> Set flag', default=None)
    parser.add_argument('-sp', '--save-path', default=None)

    args = parser.parse_args()


    main(args)    