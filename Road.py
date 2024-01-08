import gym
import numpy as np
from gym import spaces

from ObservationSpace import ObservationSpace
from utils import volume_delay_function
from functools import partial
from itertools import accumulate
from collections import defaultdict, Counter
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

    def get_unique_obs(self):
        return np.array(self.obs_fact.get_obs_unique(), dtype=np.float32)

    def get_common_obs(self):
        return np.array(self.obs_fact.get_obs_not_unique(), dtype=np.float32)

    def step(self, action):
        """
        normally in this function you perform the maths blah blah blah

        in here, we actually use this method to make the observation readible
        once we've done that, we do the hard work in the simulation class
        :param action:
        :return:
        """
        obs = self.get_obs()
        # reward = self.simulation.new_vehicles[
        #     self
        # ] * self.simulation.gym_get_specific_economic_cost(self)
        reward = self.simulation.roadQueueManager.roadRewards[self]
        done = False
        return obs, float(reward), done

    def reset(self):
        print("RESET called")
        pass

    def close(self):
        pass

    def get_road_travel_time(self, n_ext=0):
        return self.vdf(self.t_curr + n_ext)

    def get_accurate_road_travel_time(self, n_ext=0):
        road_queue = self.simulation.roadQueueManager.getQueue(self)
        if road_queue == [] and self.t_curr == 0:
            return self.get_road_travel_time()
        else:
            cumulative_etas = accumulate([r.currentETA for r in road_queue])
            etas = Counter(cumulative_etas)
            """
            adjusted_etas is basically {offset timestep from now: number of cars on road currently}
            
            So we somehow have to find out how long it would take to travel at this point
            
            """
            # breakpoint()
            adjusted_etas = {k - self.simulation.current_timestep: self.t_curr - v for k,v in etas.items()}
            # breakpoint()
            previous_value = 0
            for x in range(int(max(adjusted_etas.keys()))):
                if x not in adjusted_etas.keys():
                    adjusted_etas[x] = previous_value
                else:
                    previous_value = adjusted_etas[x]

            arrived = False
            projected_eta = 0
            travel_time = self.vdf(self.t_curr)

            while not arrived:
                print(self, self.t_curr, projected_eta, travel_time)
                if projected_eta > travel_time:
                    arrived = True
                    return self.simulation.current_timestep + projected_eta
                else:
                    projected_eta = self.vdf(adjusted_etas[projected_eta])
                    projected_eta += 1



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
