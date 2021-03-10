from utils.environment import load_environment_editor
from mlagents_envs.environment import UnityEnvironment
import numpy as np
from matplotlib import pyplot as plt
import gym
from imitation.algorithms import adversarial
from imitation.data import types


env = load_environment_editor()

data = []
env.reset()
done = False
try:
    while True:
        if done:
            done = False
        behaviour_names = list(env.behavior_specs.keys())
        behaviour_specs = list(env.behavior_specs.values())
        for name in behaviour_names:
            str_split = name.split("?")
            spec = env.behavior_specs[name]
            decision_steps, terminal_steps = env.get_steps(behavior_name=name)
            if str_split[0] == "Player":
                if len(decision_steps) == 1:
                    for agent_id in decision_steps.agent_id:
                        obs = decision_steps[agent_id].obs
                        image = obs[0]
                        obs_scalar = obs[1][0:6]
                        action = obs[1][6]
                        transition = [image, obs_scalar, action]
                        data.append(transition)
                if len(terminal_steps) == 1:
                    done = True
                    for agent_id in terminal_steps.agent_id:
                        obs = terminal_steps[agent_id].obs
                        image = obs[0]
                        obs_scalar = obs[1][0:6]
                        action = obs[1][6]
                        transition = [image, obs_scalar, action]
                        data.append(transition)
        env.step()
except:
    print("Game closed")
np.save('data.npy', data)
env.close()
# TOdo ver o ficheiro se já existe
