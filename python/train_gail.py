from utils.trainer import Trainer
from utils.gail import GAIL
from utils.buffer import RolloutBuffer
from utils.environment import load_environment, load_environment_editor
import numpy as np
import torch
from torch import cuda
from datetime import datetime
import os
if cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

demonstrations = np.load('data.npy', allow_pickle=True)
image_dim = (60, 80, 1)
obs_shape = (6,)
action_shape = (6,)
buffer_exp = RolloutBuffer(
    len(demonstrations), image_dim, obs_shape, action_shape, device)

for i in range(len(demonstrations)):
    buffer_exp.append(demonstrations[i][0],
                      demonstrations[i][1], demonstrations[i][2], 0, 0, 0, demonstrations[i][3], demonstrations[i][4])


algo = GAIL(buffer_exp=buffer_exp, image_shape=image_dim,
            state_shape=obs_shape, action_shape=action_shape, device=device, seed=int(0), rollout_length=2000)
time = datetime.now().strftime("%Y%m%d-%H%M")
log_dir = os.path.join('logs', "RedRunner", f'seed-{time}')
env = load_environment(True)
trainer = Trainer(env=env, algo=algo, log_dir=log_dir, num_steps=10**6)
trainer.train()
env.close()
