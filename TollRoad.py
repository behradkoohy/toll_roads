import gym
from gym import spaces

class TollRoad(gym.Env):

    def __init__(self):
        super(TollRoad, self).__init__()
        self.action_space = spaces.Box(low=0, high=1.0, shape=(1,))
        pass

    def step(self):
        pass

    def reset(self):
        pass

    def close(self):
        pass