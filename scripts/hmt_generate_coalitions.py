import itertools
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

random.seed(52932)
letters = ['s', 'm', 'l']
combination_length = 4


torque_options = ["LT", "MT", "HT"]
power_options = ["LP", "mP"]
torque_ranges = {
        'LT' :[5.0, 10.66],
        'MT' :[10.66, 16.32],
        'HT': [16.32, 21.98]}

power_ranges = {
        'LT' :[1.5, 2.25],
        'MT' :[2.25, 3.0]}
combinations = list(itertools.product(torque_options, power_options))

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
