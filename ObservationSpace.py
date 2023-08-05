from itertools import chain

import numpy as np


class ObservationSpace:
    def __init__(
        self, simulation, road, n_upcoming=5, n_increasing_travel_time_window=5,
    ):
        self.simulation = simulation
        self.road = road
        """
        LOW must be a list of float32
        HIGH must be a list of float32
        funct must be a list of partial or uncalled functions
        title must be a list of strings
        """
        self.obs_dict = [
            {
                "low": [
                    0 for _ in (self.simulation.toll_roads + self.simulation.free_roads)
                ],
                "high": [np.Inf for _ in self.simulation.toll_roads]
                + [0 for _ in self.simulation.free_roads],
                "funct": self.simulation.gym_get_econ_cost,
                "title": [
                    "Toll Agent Price" + str(i)
                    for i, _ in enumerate(self.simulation.toll_roads)
                ]
                + [
                    "Free Agent " + str(i + len(self.simulation.toll_roads))
                    for i, _ in enumerate(self.simulation.free_roads)
                ],
            },
            {
                "low": [
                    0 for _ in (self.simulation.toll_roads + self.simulation.free_roads)
                ],
                "high": [
                    1 for _ in (self.simulation.toll_roads + self.simulation.free_roads)
                ],
                "funct": lambda: [
                    x / self.simulation.n_cars for x in self.simulation.gym_get_demand()
                ],
                "title": [
                    "Demand Agent as ratio of total" + str(i)
                    for i, _ in enumerate(
                        self.simulation.toll_roads + self.simulation.free_roads
                    )
                ],
            },
            {
                "low": [
                    0 for _ in (self.simulation.toll_roads + self.simulation.free_roads)
                ],
                "high": [
                    1 for _ in (self.simulation.toll_roads + self.simulation.free_roads)
                ],
                "funct": lambda: [
                    x / r.c
                    for x, r in zip(
                        self.simulation.gym_get_demand(),
                        (self.simulation.toll_roads + self.simulation.free_roads),
                    )
                ],
                "title": [
                    "Demand Agent as ratio of t_c" + str(i)
                    for i, _ in enumerate(
                        self.simulation.toll_roads + self.simulation.free_roads
                    )
                ],
            },
            {
                "low": [0],
                "high": [1],
                "funct": lambda: [
                    self.simulation.gym_get_vehicles_remaining()
                    / self.simulation.n_cars
                ],
                "title": ["Number of Vehicles Remaining"],
            },
            {
                "low": [0],
                "high": [1],
                "funct": lambda: [
                    self.simulation.gym_get_same_destination(self.road.destination)
                ],
                "title": ["Cars with same destination as this road"],
            },
            {
                "low": [0],
                "high": [1],
                "funct": lambda: [
                    self.simulation.gym_get_diff_destination(self.road.destination)
                ],
                "title": ["Cars with same destination as this road"],
            },
            {
                "low": [
                    0 for _ in (self.simulation.toll_roads + self.simulation.free_roads)
                ],
                "high": [
                    np.Inf
                    for _ in (self.simulation.toll_roads + self.simulation.free_roads)
                ],
                "funct": self.simulation.gym_get_road_travel_times,
                "title": [
                    "Travel Time Agent " + str(i)
                    for i, _ in enumerate(
                        self.simulation.toll_roads + self.simulation.free_roads
                    )
                ],
            },
            {
                "low": [0],
                "high": [1],
                "funct": lambda: [self.road.get_a()],
                "title": ["road A value"],
            },
            {
                "low": [0],
                "high": [5],
                "funct": lambda: [self.road.get_b()],
                "title": ["road B value"],
            },
            {
                "low": [0],
                "high": [1],
                "funct": lambda: [self.road.get_c()],
                "title": ["road c value"],
            },
            {
                "low": [0],
                "high": [1],
                "funct": lambda: [self.road.get_t0()],
                "title": ["road t0"],
            },
            {
                "low": [0 for _ in range(n_upcoming)],
                "high": [np.Inf for _ in range(n_upcoming)],
                "funct": lambda: [
                    len(
                        self.simulation.cars.get(
                            self.simulation.current_timestep + t, []
                        )
                    )
                    for t in range(1, n_upcoming + 1)
                ],
                "title": ["cars upcoming n=" + str(t) for t in range(n_upcoming)],
            },
            {
                "low": [0 for _ in range(n_increasing_travel_time_window)],
                "high": [np.Inf for _ in range(n_increasing_travel_time_window)],
                "funct": lambda: [
                    len(
                        self.simulation.cars.get(
                            self.simulation.current_timestep + t, []
                        )
                    )
                    for t in range(n_increasing_travel_time_window)
                ],
                "title": [
                    "increase in travel time n=" + str(t)
                    for t in range(1, n_increasing_travel_time_window + 1)
                ],
            },
            {
                "low": [0],
                "high": [self.simulation.n_cars],
                "funct": lambda: [
                    len(
                        self.simulation.t_curr_adj.get(
                            self.simulation.current_timestep, []
                        )
                    )
                ],
                "title": ["Cars arriving at current timestep"],
            },
        ]

    def get_lows(self):
        return list(chain.from_iterable([x["low"] for x in self.obs_dict]))

    def get_highs(self):
        return list(chain.from_iterable([x["high"] for x in self.obs_dict]))

    def get_titles(self):
        return list(chain.from_iterable([x["title"] for x in self.obs_dict]))

    def get_obs(self):
        return list(chain.from_iterable([x["funct"]() for x in self.obs_dict]))
