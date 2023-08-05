import gym
import numpy as np
from gym import spaces

from ObservationSpace import ObservationSpace
from utils import volume_delay_function
from functools import partial
from collections import defaultdict
from numpy.random import normal, randint, beta


class DynamicRoad:
    def __init__(self):
        self.cost_update = defaultdict(int)


"""
### Action Space
The action space is a continuous variable which is the price that the road users pay.
Let's represent this with a single dimensional box space.
Min: 0
Max: 10

### Observation Space
Observation Space is a multidimensional box space, representing different variables we can use.

"""


class TollRoad(gym.Env, DynamicRoad):
    def __init__(
        self, simulation, origin, destination, max_price=10.0,
    ):
        super(TollRoad, self).__init__()
        self.simulation = simulation
        self.origin = origin
        self.destination = destination
        # self.action_space = spaces.Box(low=0.0, high=max_price, shape=(1,))
        self.action_space = spaces.Discrete(3)
        self.t_curr = 0
        self.profit = 0

    def set_obs_space(self):
        self.obs_fact = ObservationSpace(self.simulation, self)
        self.observation_space = spaces.Box(
            low=np.array(self.obs_fact.get_lows(), dtype=np.float32),
            high=np.array(self.obs_fact.get_highs(), dtype=np.float32),
            dtype=np.float32,
        )

    def reset_values(self, a, b, c, t0):
        self.a = a
        self.b = b
        self.c = c
        self.t0 = t0
        self.vdf = partial(volume_delay_function, self.a, self.b, c, self.t0)
        self.t_curr = 0
        self.profit = 0

    def get_a(self):
        return self.a

    def get_b(self):
        return self.b

    def get_c(self):
        return self.c

    def get_t0(self):
        return self.t0

    def get_t_cur(self):
        return self.t_curr

    def get_profit(self):
        return self.profit

    def get_demand(self):
        return self.simulation.get_demand()

    def get_obs(self):
        return np.array(self.obs_fact.get_obs(), dtype=np.float32)

    def step(self, action):
        """
        normally in this function you perform the maths blah blah blah

        in here, we actually use this method to make the observation readible
        once we've done that, we do the hard work in the simulation class
        :param action:
        :return:
        """
        obs = self.get_obs()
        reward = self.simulation.new_vehicles[
            self
        ] * self.simulation.gym_get_specific_economic_cost(self)
        done = False
        return obs, float(reward), done

    def reset(self):
        print("RESET called")
        pass

    def close(self):
        pass

    def get_road_travel_time(self, n_ext=0):
        return self.vdf(self.t_curr + n_ext)

    # def __repr__(self):
    #     return (
    #         str(self.simulation.gym_get_specific_economic_cost(self)) + ", " + self.t0
    #     )


class FreeRoad(DynamicRoad):
    def __init__(self, travel_time=100):
        self.travel_time = travel_time
        self.origin = None
        self.t_curr = 0
        self.a = normal(0.15, 0.1, 1)[0]
        self.b = normal(4, 1, 1)[0]
        self.c = 100
        self.t0 = travel_time
        self.vdf = partial(volume_delay_function, self.a, self.b, self.c, self.t0)

    def get_road_travel_time(self, n_ext=0):
        return self.vdf(self.t_curr + n_ext)

    # def __repr__(self):
    #     return "0, " + str(self.travel_time)
