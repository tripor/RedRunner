from utils.environment import load_environment_editor
import numpy as np


env = load_environment_editor()

data = []
env.reset()
done = False
first_on_episode = True
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
                if len(terminal_steps) == 1:
                    done = True
                    for agent_id in terminal_steps.agent_id:
                        obs = terminal_steps[agent_id].obs
                        image = obs[0]
                        obs_scalar = obs[1][0:6]
                        action = obs[1][6]
                        transition = [image, obs_scalar, action]
        if first_on_episode:
            first_on_episode = False
        else:
            transition = [last_image, last_obs, last_action, image, obs_scalar]
            data.append(transition)
        last_image = image
        last_obs = obs_scalar
        last_action = action
        env.step()
        if done:
            first_on_episode = True
except:
    print("Game closed")
np.save('data.npy', data)
env.close()
# TOdo ver o ficheiro se já existe
