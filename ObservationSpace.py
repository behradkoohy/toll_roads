from itertools import chain

import numpy as np


class ObservationSpace:
    def __init__(
        self,
        simulation,
        road,
        n_upcoming=30,
        n_increasing_travel_time_window=30,
    ):
        self.simulation = simulation
        self.road = road
        """
        LOW must be a list of float32
        HIGH must be a list of float32
        funct must be a list of partial or uncalled functions
        title must be a list of strings
        uta must be a boolean, stands for unique to agent, prevents duplicate calls to values that dont change
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
                "uta": False,
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
                "uta": False,
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
                "uta": False,
            },
            {
                "low": [0],
                "high": [1],
                "funct": lambda: [
                    self.simulation.gym_get_vehicles_remaining()
                    / self.simulation.n_cars
                ],
                "title": ["Number of Vehicles Remaining"],
                "uta": False,
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
                "uta": False,
            },
            {
                "low": [0],
                "high": [1],
                "funct": lambda: [self.road.get_a()],
                "title": ["road A value"],
                "uta": True,
            },
            {
                "low": [0],
                "high": [5],
                "funct": lambda: [self.road.get_b()],
                "title": ["road B value"],
                "uta": True,
            },
            {
                "low": [0],
                "high": [1],
                "funct": lambda: [self.road.get_c()],
                "title": ["road c value"],
                "uta": True,
            },
            {
                "low": [0],
                "high": [1],
                "funct": lambda: [self.road.get_t0()],
                "title": ["road t0"],
                "uta": True,
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
                "uta": False,
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
                "uta": False,
            },
            {
                "low": [0],
                "high": [self.simulation.n_cars],
                "funct": lambda: [len(self.simulation.arrived_vehicles)],
                "title": ["Cars arriving at current timestep"],
                "uta": False,
            },
            {
                "low": [0],
                "high": [self.simulation.n_cars],
                "funct": lambda: [self.simulation.roadQueueManager.getCarsOnRoad(road)],
                "title": ["cars queued at current timestep on this road"],
                "uta": True,
            },
            {
                "low": [0],
                "high": [self.simulation.n_cars],
                "funct": lambda: [
                    self.simulation.roadQueueManager.getTimeStepsUntilTimeReduction(
                        road
                    )
                ],
                "title": ["Timesteps until time reduction"],
                "uta": True,
            },
            {
                "low": [0],
                "high": [self.simulation.timesteps],
                "funct": lambda: [
                    self.simulation.timesteps - self.simulation.current_timestep
                ],
                "title": ["Timesteps left in simulation"],
                "uta": False,
            },
            {
                "low": [0 for _ in range(8)],
                "high": [np.inf for _ in range(8)],
                "funct": lambda: self.simulation.gym_get_arrived_car_details(),
                "title": [
                    "min arrived vot",
                    "q1 arrived vot",
                    "median arrived vot",
                    "mean arrived vot",
                    "var arrived vot",
                    "q3 arrived vot",
                    "max arrived vot",
                    "ptp arrived vot",
                ],
                "uta": False,
            },
        ]

    def get_lows(self):
        # return list(chain.from_iterable([x["low"] for x in self.obs_dict]))
        return list(
            chain.from_iterable([x["low"] for x in self.obs_dict if x["uta"] is True])
        ) + list(
            chain.from_iterable(
                [x["low"] for x in self.obs_dict if x["uta"] is False]
            )
        )

    def get_highs(self):
        # return list(chain.from_iterable([x["high"] for x in self.obs_dict]))
        return list(
            chain.from_iterable(
                [x["high"] for x in self.obs_dict if x["uta"] is True]
            )
        ) + list(
            chain.from_iterable(
                [x["high"] for x in self.obs_dict if x["uta"] is False]
            )
        )

    def get_titles(self):
        # return list(chain.from_iterable([x["title"] for x in self.obs_dict]))
        return list(
            chain.from_iterable(
                [x["title"] for x in self.obs_dict if x["uta"] is True]
            )
        ) + list(
            chain.from_iterable(
                [x["title"] for x in self.obs_dict if x["uta"] is False]
            )
        )

    def get_obs(self):
        # return list(chain.from_iterable([x["funct"]() for x in self.obs_dict]))
        return self.get_obs_unique() + self.get_obs_not_unique()

    def get_obs_unique(self):
        return list(
            chain.from_iterable(
                [x["funct"]() for x in self.obs_dict if x["uta"] is True]
            )
        )

    def get_obs_not_unique(self):
        return list(
            chain.from_iterable(
                [x["funct"]() for x in self.obs_dict if x["uta"] is False]
            )
        )
