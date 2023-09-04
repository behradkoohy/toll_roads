from pfrl import replay_buffers, explorers, q_functions
from pfrl.agents import DoubleDQN, CategoricalDQN
from pfrl.q_functions import DiscreteActionValueHead
from torch import nn, optim
import numpy as np


class DQNWrapper:
    def __new__(self, obs_size, n_epochs, timesteps, sim, n_buf=50000, *args, **kwargs):
        print("obs size", obs_size)
        self.model = nn.Sequential(
            # nn.Conv2d(obs_size, 64, kernel_size=(2, 2)),
            # nn.ReLU(),
            # nn.Flatten(),
            nn.Linear(obs_size, 32),
            nn.ReLU(),
            nn.Linear(32, 256),
            nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Linear(256, 512),
            # nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            DiscreteActionValueHead(),
        )
        print(self, (sum([len(x) for x in self.model.parameters()])))

        self.opt = optim.Adam(self.model.parameters())

        # self.replay_buffer = replay_buffers.ReplayBuffer(n_buf)
        betasteps = timesteps / 50
        replay_size = int(timesteps * n_epochs * 0.9)
        replay_alpha0 = 0.6
        replay_beta0 = 0.4
        self.replay_buffer = replay_buffers.PrioritizedReplayBuffer(
            # n_buf, alpha=0.6, beta0=0.4, betasteps=betasteps, num_steps=1
            replay_size,
            alpha=replay_alpha0,
            beta0=replay_beta0,
            betasteps=betasteps,
            num_steps=1,
        )

        decay_timestep = int(timesteps * n_epochs * 0.9)
        explr_start = 1.0
        explr_end = 0.01
        self.explorer = explorers.LinearDecayEpsilonGreedy(
            explr_start,
            explr_end,
            # 0.99,
            decay_timestep,
            lambda: np.random.randint(3),
            # 1.0,
            # 0.1,
            # 1000000,
            # lambda: np.random.randint(3),
        )
        sim.manifestmaker.write_model_manifest({
            'model_arch': [str(m) for m in self.model],
            # 'model_params': self.model.parameters(),
            'replay_size': replay_size,
            'replay_alpha0': replay_alpha0,
            'replay_beta0': replay_beta0,
            'replay_beta_steps': betasteps,
            'explorer_decay_end': decay_timestep,
            'explorer_start': explr_start,
            'explorer_end': explr_end
        })

        print("replay size:", replay_size, ", decay timestep:", decay_timestep)
        self.model = DoubleDQN(
            self.model,
            self.opt,
            self.replay_buffer,
            0.99,
            self.explorer,
            minibatch_size=2,
            replay_start_size=64,
            target_update_interval=50,
        )
        return self

    def get_model(self):
        return self.model

    def act(self, obs):
        threshold = 0.25

        act_change = self.model().act(obs) - 1
        act_funct = {
            -1: lambda x: x - threshold,
            0: lambda x: x,
            1: lambda x: x + threshold,
        }

        return act_funct[act_change]

    def observe(self, obs, reward, done, reset):
        self.model.observe(obs, reward, done, reset)



class CDQNWrapper:
    def __new__(self, obs_size, n_epochs, timesteps, n_buf=50000, *args, **kwargs):
        # self.model = nn.Sequential(
        #     # nn.Conv2d(obs_size, 64, kernel_size=(2, 2)),
        #     # nn.ReLU(),
        #     # nn.Flatten(),
        #     nn.Linear(obs_size, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 256),
        #     nn.ReLU(),
        #     # nn.Linear(256, 256),
        #     # nn.ReLU(),
        #     # nn.Linear(256, 512),
        #     # nn.ReLU(),
        #     # nn.Linear(512, 256),
        #     # nn.ReLU(),
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 3),
        #     DiscreteActionValueHead(),
        # )
        self.model = q_functions.DistributionalFCStateQFunctionWithDiscreteAction(
            obs_size, 3, 51, 0, 2, n_hidden_channels=12, n_hidden_layers=3
        )

        print(self, (sum([len(x) for x in self.model.parameters()])))

        self.opt = optim.Adam(self.model.parameters())

        # self.replay_buffer = replay_buffers.ReplayBuffer(n_buf)
        betasteps = timesteps / 50
        self.replay_buffer = replay_buffers.PrioritizedReplayBuffer(
            # n_buf, alpha=0.6, beta0=0.4, betasteps=betasteps, num_steps=1
            int(timesteps * n_epochs * 0.9),
            alpha=0.8,
            beta0=0.4,
            betasteps=betasteps,
            num_steps=1,
        )

        self.explorer = explorers.LinearDecayEpsilonGreedy(
            1.0,
            0.01,
            int(timesteps * n_epochs * 0.9),
            lambda: np.random.randint(3),
            # 1.0,
            # 0.1,
            # 1000000,
            # lambda: np.random.randint(3),
        )

        return CategoricalDQN(
            self.model,
            self.opt,
            self.replay_buffer,
            0.99,
            self.explorer,
            minibatch_size=2,
            replay_start_size=64,
            target_update_interval=50,
        )