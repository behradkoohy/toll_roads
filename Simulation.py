import codecs
import pickle
import random
import argparse

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from pfrl import explorers, replay_buffers
from pfrl.policies import GaussianHeadWithFixedCovariance
from pfrl.q_functions import DiscreteActionValueHead

from Car import Car
from Road import TollRoad, FreeRoad
from agent_config import agent_configs
from utils import agent_probability_function
from MapClasses import Origin, Destination
from Logger import Logging, ManifestMaker
from numpy.random import lognormal, normal, randint, beta
from numpy import linspace
from functools import partial
from collections import defaultdict
from itertools import combinations, product
from QueueManagement import TravellingCar, RoadQueueManager
from tqdm import trange

from Model import DQNWrapper, CDQNWrapper

from pfrl.agents import A2C, DQN, DoubleDQN
from pfrl.optimizers import SharedRMSpropEpsInsideSqrt
from torch import nn, from_numpy, optim


class DummyRL:
    def __init__(self):
        pass

    def select_action(self):
        return random.randint(0, 10)


class Simulation:
    def __init__(
        self,
        n_cars,
        n_timesteps,
        n_toll_roads=2,
        n_free_road=0,
        random_seed=0,
        n_origins=2,
        n_destinations=2,
        n_epochs=1,
        log_dir="",
        agent="",
        fixed_capacity=150,
    ):
        self.n_cars = n_cars
        self.n_free_road = n_free_road
        self.n_toll_road = n_toll_roads
        self.arrived_vehicles = []
        self.toll_roads = []
        self.timesteps = n_timesteps
        self.log_dir = log_dir
        self.log = Logging(db_file=log_dir + "logging.db")
        self.manifestmaker = ManifestMaker(log_dir)
        self.agent_inp = agent
        self.fixed_capacity = fixed_capacity
        # self.car_dist_arrival = [round(x) for x in linspace(0, n_timesteps, n_cars)]
        beta_dist_alpha = 5
        beta_dist_beta = 5

        self.car_dist_norm = beta(beta_dist_alpha, beta_dist_beta, size=self.n_cars)
        self.car_dist_arrival = list(
            map(
                lambda z: round(
                    (z - min(self.car_dist_norm))
                    / (max(self.car_dist_norm) - min(self.car_dist_norm))
                    * self.timesteps
                ),
                self.car_dist_norm,
            )
        )
        self.car_dist_deadline = sorted(randint(4, self.timesteps, n_cars))
        self.origins = [Origin(x) for x in range(n_origins)]
        self.destinations = [Destination(x) for x in range(n_destinations)]
        self.toll_roads = []
        self.free_roads = []
        self.agents = {}

        for x in range(self.n_toll_road):
            road = TollRoad(self, self.origins[x], self.destinations[x])
            self.toll_roads.append(road)

        for x in range(self.n_free_road):
            road = FreeRoad(self)
            self.free_roads.append(road)

        self.manifestmaker.create_model_manifest()

        for road in self.toll_roads:
            road.set_obs_space()
            obs_size = len(road.obs_fact.get_lows())
            self.agents[road] = agent_configs[agent]["agent"](
                obs_size, n_epochs, n_timesteps, self
            )

        self.log.set_titles(self.toll_roads[0].obs_fact.get_titles())

        self.manifestmaker.write_simulation_manifest(
            {
                "n_cars": self.n_cars,
                "n_timesteps": self.timesteps,
                "n_free_roads": self.n_free_road,
                "n_toll_roads": self.n_toll_road,
                "n_origins": n_origins,
                "n_destinations": n_destinations,
                "n_epochs": n_epochs,
                "beta_dist_alpha": beta_dist_alpha,
                "beta_dist_beta": beta_dist_beta,
                "obs_size": len(road.obs_fact.get_titles()),
                "obs_titles": road.obs_fact.get_titles(),
            }
        )

        self.roadQueueManager = RoadQueueManager(self)

        for self.epoch in trange(n_epochs, unit="epochs"):
            self.build_environment(n_origins, n_destinations)
            self.start_sim()
        self.log.end()

    def build_environment(self, n_origins, n_destinations):
        # random.seed(1)
        self.cars = defaultdict(list)  # {timestep of arrival: car details}

        a_dist = normal(
            0.15, 0.1, self.n_toll_road
        )  # TODO: this can output negative numbers, fix before implementing
        b_dist = normal(4, 1, self.n_toll_road)
        # self.road_cost = {r: 15 for r in self.toll_roads}
        self.road_cost = {}
        # generate the toll roads and their values
        # road_ct0 = [(10, 10), (10, 10)]
        # road_ct0 = [(9999, 60), (9999, 20)]
        road_ct0 = [(self.fixed_capacity, 10), (self.fixed_capacity, 10)]
        for rd in zip(enumerate(self.toll_roads), road_ct0):
            x = rd[0][0]
            road = rd[0][1]
            c = rd[1][0]
            t0 = rd[1][1]
            self.road_cost[rd[0][1]] = rd[1][1]
            # self.road_cost[rd[0][1]] = 30
            # c = random.randint(10, 30)
            # c = 20
            # t0 = 15
            road.reset_values(a_dist[x], b_dist[x], c, t0)
            # print("Toll road:", c, t0, "a, b set to 0.15, 4")
        # generate the free roads
        self.free_roads = [
            # FreeRoad(randint(15, round(self.timesteps * 0.9)))
            FreeRoad(200)
            for _ in range(self.n_free_road)
        ]

        self.roadQueueManager.clearQueue()

        # initialise epsilon of o and d
        # epsilon is simply the extra distance to go from one destination to the other
        # i.e. you are at d1 but you are meant to get to d2
        # these costs are static
        # self.epsilon_o = randint(0, 5)
        # self.epsilon_d = randint(0, 5)
        self.epsilon_o = 0
        self.epsilon_d = 0

        self.route_matrix = {}
        for o, d, r in zip(self.origins, self.destinations, self.toll_roads):
            self.route_matrix[(o, d, r)] = r.t0

        vehicle_vot = np.random.uniform(2.5, 9.5, self.n_cars)
        # vehicle_vot = np.asarray([1 for x in range(self.n_cars)])

        car_details = list(
            zip(
                [None for x in range(self.n_cars)],
                [None for x in range(self.n_cars)],
                self.car_dist_arrival,
                [
                    (d if a < d else a + 1)
                    for d, a in zip(self.car_dist_deadline, self.car_dist_arrival)
                ],
                vehicle_vot,
            )
        )
        # print("Generating car objects")
        for n in range(self.timesteps + 1):
            timestep_vehicles = [x for x in car_details if x[2] == n]
            for vehicle in timestep_vehicles:
                self.cars[n].append(
                    Car(
                        *vehicle,
                        random.sample(self.origins, 1)[0],  # Vehicle origin
                        random.sample(self.destinations, 1)[0],  # Vehicle destination
                        # random.randint(0, 150)  # Vehicle budget
                    )
                )
        # print(self.road_cost)

    def get_demand(self):
        return [r.t_curr for r in self.toll_roads + self.free_roads]

    def get_roads(self):
        return self.toll_roads + self.free_roads

    def set_toll_road_price(self, road, price):
        self.road_cost[road] = price

    def get_road_economic_cost(self):
        return [x for x in self.road_cost.values()]

    def update_road_from_decsion(self, decision):
        """
        :param decision: this is just the road that is chosen
        :return: None
        """
        # road = [r for r in (self.toll_roads + self.free_roads) if r is decision]
        decision[0][2].t_curr += 1

    def gym_get_econ_cost(self):
        return self.get_road_economic_cost() + [0 for _ in self.free_roads]

    def gym_get_demand(self):
        return [r.t_curr for r in self.toll_roads + self.free_roads]

    def gym_get_specific_economic_cost(self, road):
        return self.road_cost[road]


    def gym_get_vehicles_remaining(self):
        vehicles_left = self.n_cars - self.roadQueueManager.arrived_vehicles
        return vehicles_left

    def gym_get_road_travel_times(self):
        return [
            road.get_road_travel_time() for road in self.toll_roads + self.free_roads
        ]

    def gym_get_arrived_car_details(self):
        cars = self.arrived_vehicles
        if cars == []:
            return [0,0,0,0,0,0,0,0]
        else:
            car_vots = [c.vot for c in cars]
            return [
                np.min(car_vots),
                np.quantile(car_vots, 0.25),
                np.median(car_vots),
                np.average(car_vots),
                np.var(car_vots),
                np.quantile(car_vots, 0.75),
                np.max(car_vots),
                np.ptp(car_vots)
            ]

    """
    TODO: 
        1. number of vehicles arriving at this timestep DONE
        2. number of cars arriving next timestep (1-10) DONE (up to 30)
        3. Mean, Q1, Q3, Median, Min, Max VoT of vehicles arrived at this timestep DONE
        4. cars in current road queue DONE
        5. cars in other road queue NOT FEASIBLE EASILY
        6. timesteps until queue reduces DONE
        7. timesteps left in simulation DONE
    """

    def start_sim(self):
        self.threshold = 0.25
        total_reward = defaultdict(int)
        self.current_timestep = 0
        while not self.roadQueueManager.isSimulationComplete():
            # First, we update the agents actions
            common_obs = self.toll_roads[0].get_common_obs()
            for road in self.toll_roads:
                unique_obs = road.get_unique_obs()
                obs = np.concatenate([common_obs, unique_obs])
                act = self.agents[road].model.act(from_numpy(obs).type(torch.float32))
                new_price = self.gym_get_specific_economic_cost(road) + (act - 1)
                if new_price < self.threshold:
                    self.set_toll_road_price(road, self.threshold)
                else:
                    self.set_toll_road_price(road, new_price)

            # Next, we update the number of vehicles that have arrived at this timestep
            self.arrived_vehicles = self.cars[self.current_timestep]
            # if self.arrived_vehicles == []:
                # continue
            self.roadQueueManager.updateQueue()
            road_travel_time = {
                road: road.get_road_travel_time()
                for road in self.toll_roads + self.free_roads
            }
            road_econom_cost = {
                road: self.road_cost.get(road, 0)
                for road in self.toll_roads + self.free_roads
            }
            # this is the hard bit, I want to generate and make the quantal process without a while loop
            roadUtilityFunct = {
                road: lambda x: (
                    -((x * road_travel_time[road]) + road_econom_cost[road])
                )
                for road in self.toll_roads + self.free_roads
            }
            # TODO: go from the line above into a functioning simulator lol
            # TODO: I think the key will be to use a loop atleast, currently missing one step and not sure what it is
            decisions = [
                list(
                    map(
                        car.new_quantal_decision,
                        [[
                            (road, roadfn(car.vot))
                            for road, roadfn in roadUtilityFunct.items()
                        ]]
                    )
                )
                for car in self.arrived_vehicles
            ]

            for decision, car in zip(decisions, self.arrived_vehicles):
                # breakpoint()
                tc = TravellingCar(
                    car,
                    hash(car),
                    self.current_timestep,
                    self.current_timestep + road_travel_time[decision[0][0]],
                    self.current_timestep + road_travel_time[decision[0][0]],
                    decision[0],
                    car.vot
                )
                self.roadQueueManager.addToQueue(decision[0][0], tc)

            for road in self.toll_roads:
                obs, reward, done = road.step(act)
                total_reward[road] += reward
                self.agents[road].observe(
                    self.agents[road],
                    obs,
                    reward,
                    self.current_timestep == self.timesteps + 1,
                    self.current_timestep == self.timesteps + 1,
                )
                self.roadQueueManager.clearRewards()
            self.current_timestep += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="MMRP Simulation")
    ap.add_argument("-C", "--cars", default=300, action="store", type=int)
    ap.add_argument("-T", "--timesteps", default=200, action="store", type=int)
    ap.add_argument("-E", "--epochs", default=100, action="store", type=int)
    ap.add_argument("-TR", "--tollroads", default=2, action="store", type=int)
    ap.add_argument("-FR", "--freeroads", default=0, action="store", type=int)
    ap.add_argument("-L", "--logdir", default="./", action="store", type=str)
    ap.add_argument(
        "-A",
        "--agent",
        default="Random",
        choices=[agent for agent in agent_configs],
        type=str,
    )
    ap.add_argument("-CP", "--capacity", default=100, action="store", type=int)
    a = ap.parse_args()
    print(a)
    # # for x in range(10):
    # cars = 300
    # timesteps = 200
    # epochs = 100
    # print("Number of training timesteps", timesteps * epochs)
    # s = Simulation(cars, timesteps, n_epochs=epochs)
    s = Simulation(
        a.cars,
        a.timesteps,
        n_epochs=a.epochs,
        n_toll_roads=a.tollroads,
        n_free_road=a.freeroads,
        log_dir=a.logdir,
        agent=a.agent,
        fixed_capacity=a.capacity,
    )
    # s.log.conn.commit()
    # # s.log.pretty_graphs()
