import gym
import numpy as np
from gym import spaces
from utils import volume_delay_function
from functools import partial
from collections import defaultdict


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
    def __init__(self, simulation, a, b, c, t0, origin, destination, max_price=10.0):
        super(TollRoad, self).__init__()
        self.t0 = t0
        self.action_space = spaces.Box(low=0, high=max_price, shape=(1,))
        self.simulation = simulation
        n_roads = range(simulation.n_toll_road + simulation.n_free_road)
        self.observation_space = spaces.Box(
            low=np.array(
                [0 for x in n_roads]  # representing the minimum cost of every road
                + [0 for x in n_roads]  # representing the minimum demand of every road
            ),
            high=np.array(
                [max_price for x in n_roads] + [simulation.n_cars for x in n_roads]
            ),
            dtype=np.float32,
        )
        self.vdf = partial(volume_delay_function, a, b, c, self.t0)
        self.t_curr = 0
        self.origin = origin
        self.destination = destination
        self.profit = 0

    def get_demand(self):
        return self.simulation.get_demand()

    def step(self, action):
        """
        normally in this function you perform the maths blah blah blah

        in here, we actually use this method to make the observation readible
        once we've done that, we do the hard work in the simulation class
        :param action:
        :return:
        """
        econ_cost = self.simulation.gym_get_econ_cost()
        demand = self.simulation.gym_get_demand()
        obs = np.array(econ_cost + demand)
        # obs, reward, done = None
        # reward = None
        reward = self.simulation.new_vehicles[self] * self.simulation.gym_get_specific_economic_cost(self)
        done = False
        return obs, reward, done

    def reset(self):
        pass

    def close(self):
        pass

    def get_road_travel_time(self):
        return self.vdf(self.t_curr)


class FreeRoad(DynamicRoad):
    def __init__(self, travel_time):
        self.travel_time = travel_time
        self.origin = None
        self.t_curr = 0

    def get_road_travel_time(self):
        return self.travel_time
