import os
from time import time, sleep
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, env, algo, log_dir, seed=0, num_steps=10**5,
                 eval_interval=10**3, num_eval_episodes=5):
        super().__init__()

        # Env to collect samples.
        self.env = env

        self.algo = algo
        self.log_dir = log_dir

        # Log setting.
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset()
        image = None
        state = None
        action = None
        done = False
        step = 0
        iteration = 0
        while step < self.num_steps + 1:
            # Pass to the algorithm to update state and episode timestep.
            image, state, action, done, t = self.algo.step(
                self.env, image, state, action, done, t, step)

            # Update the algorithm whenever ready.
            if step != 0 and self.algo.is_update(step):
                iteration += 1
                print("Iteration {}".format(iteration))
                self.algo.update(self.writer)
            if not done:
                step += 1
            # Wait for the logging to be finished.
        sleep(10)

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
