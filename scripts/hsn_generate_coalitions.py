import itertools
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

random.seed(52932)
letters = ['s', 'm', 'l']
combination_length = 4

combinations = list(itertools.combinations_with_replacement(letters, combination_length))

ranges = {
        's' :[0.2, 0.34],
        'm': [0.33, 0.47],
        'l' : [0.46, 0.61] }

output = {'train': {'coalitions': {f"{combination_length}_agents": {}}}}

def combination_to_traits(c):
    team = []
    for ch in c:
        team.append(random.uniform(ranges[ch][0], ranges[ch][1]))
    return(team)

print("Total combinations:", len(combinations))
print("List of combinations:")

teams = []
for combination in combinations:
    c = ''.join(combination)
    team = combination_to_traits(c)
    teams.append(team)
    print(c, " --> ", team)


for team_index, team in enumerate(teams):
    N = len(team)
    fig, ax = plt.subplots()
    output["train"]["coalitions"][f"{combination_length}_agents"][team_index] = {}
    for i in range(N):
        
        agent = {"id": format(i, '06b'), "radius": team[i]}
        output["train"]["coalitions"][f"{combination_length}_agents"][team_index][i] = agent

        x = i * 1.5  # Horizontal position of the circle
        y = 0.4  # Vertical position of the circle
        color = plt.cm.viridis(team[i] / 0.6)  # Map radii to colormap
        circle = plt.Circle((x, y), team[i], color=color)
        ax.add_artist(circle)
    ax.set_xlim(-1, (N-1) * 1.5 + 1)
    ax.set_ylim(-2, 2)
    save_path = os.path.join("coalition_visuals", str(combination_length))
    if(not os.path.exists(save_path)):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path,f"{team_index}.png"))

with open(os.path.join(save_path, "coalitions.yaml"), 'w') as outfile:
    yaml.dump(output, outfile, default_flow_style=False, allow_unicode=True)

plt.show()
