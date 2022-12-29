import random

import numpy as np

from Car import Car
from Road import TollRoad, FreeRoad
from utils import agent_probability_function
from MapClasses import Origin, Destination
from numpy.random import lognormal, normal, randint
from numpy import linspace
from functools import partial
from collections import defaultdict
from itertools import combinations, product

import logging, sys

from stable_baselines3 import A2C


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


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
    ):
        self.n_cars = n_cars
        self.n_free_road = n_free_road
        self.n_toll_road = n_toll_roads
        self.arrived_vehicles = []
        self.toll_roads = []
        a_dist = normal(
            0.15, 0.1, n_toll_roads
        )  # TODO: this can output negative numbers, fix before implmnenting
        b_dist = normal(4, 1, n_toll_roads)
        # logging.debug("A dists:%s", str(a_dist))
        # logging.debug("B dists:%s", str(b_dist))
        self.origins = [Origin(x) for x in range(n_origins)]
        self.destinations = [Destination(x) for x in range(n_destinations)]

        # generate the toll roads and their values
        for x in range(n_toll_roads):
            c = random.randint(10, 30)
            t0 = 15
            self.toll_roads.append(
                TollRoad(
                    self,
                    a_dist[x],
                    b_dist[x],
                    c,
                    t0,
                    self.origins[x],
                    self.destinations[x],
                )
            )
            print("Toll road:", c, t0, "a, b set to 0.15, 4")
        # MAX_FREE_ROAD_TRAVEL_TIME
        self.free_roads = [
            FreeRoad(randint(15, round(n_timesteps * 0.9))) for _ in range(n_free_road)
        ]
        self.timesteps = n_timesteps

        self.cars = defaultdict(list)  # {timestep of arrival: car details}

        # initialise route matrix with default values
        self.route_matrix = {}
        for o, d, r in zip(self.origins, self.destinations, self.toll_roads):
            self.route_matrix[(o, d, r)] = r.t0

        # initialise epsilon of o and d
        # epsilon is simply the extra distance to go from one destination to the other
        # i.e. you are at d1 but you are meant to get to d2
        # these costs are static
        self.epsilon_o = randint(0, 5)
        self.epsilon_d = randint(0, 5)

        self.car_dist_k = lognormal(
            0, 1, n_cars
        )  # This must be positive in all cases, represents recession rate
        self.car_dist_x0 = normal(
            1, 1, n_cars
        )  # This can be any real number, represents the value of y=0.5
        self.car_dist_arrival = [round(x) for x in linspace(0, n_timesteps, n_cars)]
        self.car_dist_deadline = sorted(randint(4, self.timesteps, n_cars))
        # self.car_dist_deadline = poisson(50, n_cars)
        car_details = list(
            zip(
                self.car_dist_k,
                self.car_dist_x0,
                self.car_dist_arrival,
                [
                    (d if a < d else a + 1)
                    for d, a in zip(self.car_dist_deadline, self.car_dist_arrival)
                ],
            )
        )
        print("Generating car objects")
        for n in range(n_timesteps + 1):
            timestep_vehicles = [x for x in car_details if x[2] == n]
            for vehicle in timestep_vehicles:
                self.cars[n].append(
                    Car(
                        *vehicle,
                        random.sample(self.origins, 1)[0],  # Vehicle origin
                        random.sample(self.destinations, 1)[0],  # Vehicle destination
                        random.randint(0, 15)  # Vehicle budget
                    )
                )
        self.road_cost = {r: 0 for r in self.toll_roads}

        self.start_sim()

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
        return self.get_road_economic_cost() + [0 for r in self.free_roads]

    def gym_get_demand(self):
        return [r.t_curr for r in self.toll_roads + self.free_roads]

    def gym_get_specific_economic_cost(self, road):
        return self.road_cost[road]




    def start_sim(self):
        # update vehicles which have arrived at the function
        agents = {r: DummyRL() for r in self.toll_roads}
        agents = {r: A2C("MlpPolicy", r) for r in self.toll_roads}
        arrived = 0

        t_curr_adj = {r: defaultdict(int) for r in (self.toll_roads + self.free_roads)}
        self.new_vehicles = defaultdict(int)

        econ_cost = self.gym_get_econ_cost()
        demand = self.gym_get_demand()
        obs = np.array(econ_cost + demand)
        total_reward = defaultdict(int)

        for t in range(self.timesteps + 2):
            for road in self.toll_roads:
                act = agents[road].predict(obs)[0].item()
                obs, reward, done = road.step(act)
                total_reward[road] += reward
                # print("previous round reward:", reward)
                self.set_toll_road_price(road, act)
            # print("Timestep", t)
            # print(
            #     "travel times:",
            #     [r.get_road_travel_time() for r in self.toll_roads + self.free_roads],
            # )
            # print(
            #     "travel cost:",
            #     self.get_road_economic_cost() + [0 for r in self.free_roads],
            # )
            # add arrived vehicles at this timestep
            self.arrived_vehicles += self.cars[t]

            for road in self.toll_roads + self.free_roads:
                # print(t_curr_adj[road][t], "has arrived at destination")
                arrived += t_curr_adj[road][t]
                road.t_curr -= t_curr_adj[road][t]

            # update the class price function

            self.timestep_route_cost_vectors = {}
            #
            # for decision in self.toll_roads:
            #     self.set_toll_road_price(decision, agents[decision].select_action())
            # first, lets review the dominated pairings for all routes
            for (origin, destination) in product(self.origins, self.destinations):
                econ_cost = self.get_road_economic_cost() + [0 for _ in self.free_roads]
                time_cost = list(
                    self.get_road_time_cost(origin, destination).values()
                ) + [r.get_road_travel_time() for r in self.free_roads]
                road_id = [r for r in self.toll_roads] + [r for r in self.free_roads]

                self.timestep_route_cost_vectors[origin, destination] = list(
                    zip(econ_cost, time_cost, road_id)
                )

            vehicle_choice_list = []

            while len(self.arrived_vehicles) > 0:
                car = self.arrived_vehicles[0]
                """
                Here,
                loop through the cars, combining the travel time and cost vectors
                roll dice and allocate vehicles to them
                and bish bash bosh
                get the RL working
                """

                vehicle_specific_route_costs = self.timestep_route_cost_vectors[
                    car.origin, car.destination
                ]
                # first, lets remove any values which are over the car's budget
                vehicle_specific_route_costs = [
                    (e, t, r)
                    for (e, t, r) in vehicle_specific_route_costs
                    if e <= car.budget
                ]
                # lets look for dominated conditions and remove them
                for (cond1, cond2) in combinations(vehicle_specific_route_costs, 2):
                    if (
                        cond1 not in vehicle_specific_route_costs
                        or cond2 not in vehicle_specific_route_costs
                    ):
                        continue
                    e1, c1, r1 = cond1
                    e2, c2, r2 = cond2
                    if e1 <= e2 and c1 < c2:
                        # checking forward and backward condition due to the reflexivity of domination
                        vehicle_specific_route_costs.remove(cond2)
                    elif e1 >= e2 and c1 > c2:
                        vehicle_specific_route_costs.remove(cond1)
                # so we should never have 0 items in the costs
                if len(vehicle_specific_route_costs) == 0:
                    breakpoint()

                decision = car.make_decision(vehicle_specific_route_costs)[0]

                vehicle_choice_list.append(decision)

                self.arrived_vehicles.remove(car)


            before = [r.t_curr for r in self.toll_roads + self.free_roads]
            # print("before", [r.t_curr for r in self.toll_roads + self.free_roads])
            for decision in vehicle_choice_list:
                # if decision[0] == 0:
                #     arrived += 1
                #     continue
                decision[2].t_curr += 1
                eta = round(t + decision[2].get_road_travel_time())
                # print(eta)
                t_curr_adj[decision[2]][eta] += 1
            # print("After", [r.t_curr for r in self.toll_roads + self.free_roads])
            for (road, (old, new)) in zip(self.get_roads(), zip(before, [r.t_curr for r in self.toll_roads + self.free_roads])):
                self.new_vehicles[road] = new-old

        # offer routes to vehicle, calculate probabilities of taking each route

        # roll dice to see if route is taken
        """
        end roll 
        """
        # for road in self.toll_roads:
        #     for t in range(100, )
        print(
            "arrived",
            arrived,
            "and in queue:",
            sum([x.t_curr for x in (self.toll_roads + self.free_roads)]),
            "total:",
            (sum([x.t_curr for x in (self.toll_roads + self.free_roads)]) + arrived),
            "total rewards:\n",
            total_reward
        )


if __name__ == "__main__":
    # for x in range(10):
    s = Simulation(123, 100)


