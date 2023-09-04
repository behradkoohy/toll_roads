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
        n_free_road=1,
        random_seed=0,
        n_origins=2,
        n_destinations=2,
        n_epochs=2,
        log_dir="",
        agent="",
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
            self.agents[road] = agent_configs[agent]['agent'](obs_size, n_epochs, n_timesteps, self)

        self.log.set_titles(self.toll_roads[0].obs_fact.get_titles())

        self.manifestmaker.write_simulation_manifest({
            "n_cars" : self.n_cars,
            "n_timesteps": self.timesteps,
            "n_free_roads": self.n_free_road,
            "n_toll_roads": self.n_toll_road,
            "n_origins": n_origins,
            "n_destinations": n_destinations,
            "n_epochs": n_epochs,
            "beta_dist_alpha": beta_dist_alpha,
            "beta_dist_beta": beta_dist_beta,
            "obs_size": len(road.obs_fact.get_titles()),
            "obs_titles": road.obs_fact.get_titles()
        })

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
        road_ct0 = [(90, 60), (30, 20)]
        # road_ct0 = [(9999, 60), (9999, 20)]
        for rd in zip(enumerate(self.toll_roads), road_ct0):
            x = rd[0][0]
            road = rd[0][1]
            c = rd[1][0]
            t0 = rd[1][1]
            # self.road_cost[rd[0][1]] = rd[1][1]
            self.road_cost[rd[0][1]] = 30
            # c = random.randint(10, 30)
            # c = 20
            # t0 = 15
            road.reset_values(a_dist[x], b_dist[x], c, t0)
            # print("Toll road:", c, t0, "a, b set to 0.15, 4")
        # generate the free roads
        self.free_roads = [
            # FreeRoad(randint(15, round(self.timesteps * 0.9)))
            FreeRoad(150)
            for _ in range(self.n_free_road)
        ]

        # initialise epsilon of o and d
        # epsilon is simply the extra distance to go from one destination to the other
        # i.e. you are at d1 but you are meant to get to d2
        # these costs are static
        # self.epsilon_o = randint(0, 5)
        # self.epsilon_d = randint(0, 5)
        self.epsilon_o = 1
        self.epsilon_d = 1

        self.route_matrix = {}
        for o, d, r in zip(self.origins, self.destinations, self.toll_roads):
            self.route_matrix[(o, d, r)] = r.t0

        vehicle_vot = np.random.uniform(0.2, 2, self.n_cars)
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

    def get_road_time_cost(self, origin, destination):
        route_costs = {}
        for road in self.toll_roads:
            epsilon = 0
            if road.origin != origin:
                epsilon += self.epsilon_o
            if road.destination != destination:
                epsilon += self.epsilon_d
            route_costs[road] = road.get_road_travel_time() + epsilon

        for road in self.free_roads:
            route_costs[road] = road.get_road_travel_time()
        return route_costs

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

    def gym_get_vehicles_info(self):
        # first we need to get the number of cars left
        vehicles_left = self.n_cars - self.arrived_vehicle_count
        breakpoint()

        return self.cars

    def gym_get_vehicles_remaining(self):
        vehicles_left = self.n_cars - self.arrived_vehicle_count
        return vehicles_left

    def gym_get_same_destination(self, destination):
        return len(
            [
                i
                for sub in self.cars.values()
                for i in sub
                if (i.destination == destination)
                and (i.arrival >= self.current_timestep)
            ]
        )

    def gym_get_diff_destination(self, destination):
        return len(
            [
                i
                for sub in self.cars.values()
                for i in sub
                if (i.destination != destination)
                and (i.arrival >= self.current_timestep)
            ]
        )

    def gym_get_road_travel_times(self):
        return [
            road.get_road_travel_time() for road in self.toll_roads + self.free_roads
        ]

    def start_sim(self):
        threshold = 0.25
        # update vehicles which have arrived at the function
        arrived = 0

        self.t_curr_adj = {
            r: defaultdict(int) for r in (self.toll_roads + self.free_roads)
        }
        self.new_vehicles = defaultdict(int)
        self.arrived_vehicle_count = 0
        econ_cost = self.gym_get_econ_cost()
        # demand = self.gym_get_demand()
        # obs = np.array(econ_cost + demand, dtype="float32")
        total_reward = defaultdict(int)
        for t in range(self.timesteps + 2):
            cycle_information = {}
            self.current_timestep = t
            for n, road in enumerate(self.toll_roads):
                # self.road_time_costs = {r.get_road_travel_time(): r for r in (self.toll_roads + self.free_roads)}
                obs = road.get_obs()

                act = self.agents[road].model.act(from_numpy(obs).type(torch.float32))

                if act < threshold:
                    act = (econ_cost[n])

                cycle_information[n, "act"] = act
                cycle_information[n, "pri"] = econ_cost[n]

                obs, reward, done = road.step(act)
                cycle_information[n, "obs"] = codecs.encode(
                    pickle.dumps(obs), "base64"
                ).decode()
                cycle_information[n, "rew"] = reward

                total_reward[road] += reward
                self.agents[road].observe(
                    self.agents[road], obs, reward, t == self.timesteps + 2, t == self.timesteps + 2
                )


                self.set_toll_road_price(road, act)

            # add arrived vehicles at this timestep
            self.arrived_vehicles += self.cars[t]
            self.arrived_vehicle_count += len(self.cars[t])

            for road in self.toll_roads + self.free_roads:
                arrived += self.t_curr_adj[road][t]
                road.t_curr -= self.t_curr_adj[road][t]

            # update the class price function

            self.timestep_route_cost_vectors = {}

            # first, lets review the dominated pairings for all routes
            for (origin, destination) in product(self.origins, self.destinations):
                econ_cost = self.get_road_economic_cost() + ([0] * self.n_free_road)

                time_cost = list(
                    self.get_road_time_cost(origin, destination).values()
                ) + [r.get_road_travel_time() for r in self.free_roads]

                road_id = [r for r in self.toll_roads] + [r for r in self.free_roads]

                self.timestep_route_cost_vectors[origin, destination] = list(
                    zip(econ_cost, time_cost, road_id)
                )

            road_adj_tcur = {}
            comp_vehicles = []
            while len(self.arrived_vehicles) > 0:
                car = self.arrived_vehicles[0]
                vehicle_specific_route_costs = [
                    # (e, t, r, e / t)
                    (e, t, r, -((car.vot * t) + e))
                    for (e, t, r) in self.timestep_route_cost_vectors[
                        (car.origin, car.destination)
                    ]
                ]
                """
                NOTE: look into this
                we are using e/t to calculate the utility but we're also using car budget to limit our choices
                this isn't correct
                we need to edit budget to be value of time and to utilise that instead

                """
                # decision = car.make_decision(vehicle_specific_route_costs)[0]
                choices = car.make_quantal_decision(vehicle_specific_route_costs)
                decision = choices[0]
                # print(decision)
                # vehicle_choice_list.append(decision)
                road_adj_tcur[decision[2]] = road_adj_tcur.get(decision[2], 0) + 1

                self.t_curr_adj[decision[2]][
                    round(self.current_timestep + decision[1])
                ] = (
                    self.t_curr_adj[decision[2]][
                        round(self.current_timestep + decision[1])
                    ]
                    + 1
                )
                self.arrived_vehicles.remove(car)
                comp_vehicles.append(
                    (hash(car),
                    self.epoch,
                    self.current_timestep,
                    round(self.current_timestep + decision[1]),
                    str((decision[0], decision[2].t0)),
                    car.vot,)
                )
            self.log.batch_add_new_completed_vehicle(comp_vehicles)

            for r in self.toll_roads + self.free_roads:
                self.new_vehicles[r] = road_adj_tcur.get(r, 0)
                r.t_curr += road_adj_tcur.get(r, 0)

            logger_params = []
            for n in range(self.n_toll_road):
                for l_arg in ["obs", "act", "rew", "pri"]:
                    if l_arg == "obs":
                        logger_params.append(str(cycle_information[n, l_arg]))
                    else:
                        logger_params.append(cycle_information[n, l_arg])
            self.log.add_timestep_results(self.epoch, t, *logger_params)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="MMRP Simulation")
    ap.add_argument("-C", "--cars", default=300, action="store", type=int)
    ap.add_argument("-T", "--timesteps", default=200, action="store", type=int)
    ap.add_argument("-E", "--epochs", default=100, action="store", type=int)
    ap.add_argument("-TR", "--tollroads", default=2, action="store", type=int)
    ap.add_argument("-FR", "--freeroads", default=1, action="store", type=int)
    ap.add_argument("-L", "--logdir", default="./", action="store", type=str)
    ap.add_argument("-A", "--agent", default="DQN", choices=[agent for agent in agent_configs] , type=str)
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
        agent=a.agent
    )
    # s.log.conn.commit()
    # # s.log.pretty_graphs()
