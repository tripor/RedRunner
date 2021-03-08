from utils.environment import load_environment_editor
from mlagents_envs.environment import UnityEnvironment
import numpy as np
from matplotlib import pyplot as plt

env = load_environment_editor()

env.reset()
while True:
    behaviour_names = list(env.behavior_specs.keys())
    behaviour_specs = list(env.behavior_specs.values())
    print(behaviour_names)
    for name in behaviour_names:
        str_split = name.split("?")
        spec = env.behavior_specs[name]
        print(spec.)
        decision_steps, terminal_steps = env.get_steps(behavior_name=name)
        if str_split[0] == "Player":
            print("Player recording")
            for agent_id in decision_steps.agent_id:
                print(agent_id)
                obs = decision_steps[agent_id].obs
                print(obs)
                #image = np.array(obs[0])
    env.step()

env.close()
