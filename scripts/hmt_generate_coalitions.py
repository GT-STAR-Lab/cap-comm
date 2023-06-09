import itertools
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

def one_hot_id(idx, N=4):
    vector = [0] * N
    vector[idx] = 1
    return ''.join(str(num) for num in vector)

random.seed(52932)
letters = ['s', 'm', 'l']
combination_length = 4

combinations = list(itertools.combinations_with_replacement(letters, combination_length))
max_cap=10
ranges = {
        's' :[0, 3],
        'm': [3, 7],
        'l' : [7, 10] }

output = {'train': {'coalitions': {f"{combination_length}_agents": {}}}}

def combination_to_traits(c):
    team = []
    for ch in c:
        team.append(random.randint(ranges[ch][0], ranges[ch][1]))
    return(team)

print("Total combinations:", len(combinations))
print("List of combinations:")

teams = []
for combination in combinations:
    c = ''.join(combination)
    team = combination_to_traits(c)
    teams.append(team)
    print(c, " --> ", team)

id_val = 0
for team_index, team in enumerate(teams):
    N = len(team)
    fig, ax = plt.subplots()
    output["train"]["coalitions"][f"{combination_length}_agents"][team_index] = {}
    for i in range(N):
        
        lumber_cap = team[i]; concrete_cap = max_cap - team[i]
        agent = {"id": i, "lumber_cap": team[i], "concrete_cap": max_cap - team[i]}
        id_val += 1
        output["train"]["coalitions"][f"{combination_length}_agents"][team_index][i] = agent

        x = i * 1.5  # Horizontal position of the circle
        y = 0.4  # Vertical position of the circle
        radius = 0.4
        # color = plt.cm.viridis(radius / 0.6)  # Map radii to colormap
        color = np.array([lumber_cap, 0.1, concrete_cap]) / (max_cap+1.01)
        
        circle = plt.Circle((x, y), radius, color=color)
        ax.add_artist(circle)
    ax.set_xlim(-1, (N-1) * 1.5 + 1)
    ax.set_ylim(-2, 2)
    save_path = os.path.join("material_transport_coalition_visuals", str(combination_length))
    if(not os.path.exists(save_path)):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path,f"{team_index}.png"))

with open(os.path.join(save_path, "coalitions.yaml"), 'w') as outfile:
    yaml.dump(output, outfile, default_flow_style=False, allow_unicode=True)

plt.show()
