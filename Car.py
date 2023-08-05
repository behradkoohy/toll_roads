import math
from functools import partial

import numpy as np

from utils import agent_probability_function
from random import choices


class Car:
    def __init__(
        self, k, x0, arrival, deadline, vot, origin, destination,
    ):
        self.k = k
        self.x0 = x0
        self.determine_funct = partial(agent_probability_function, k, x0)
        self.origin = origin
        self.arrival = arrival
        self.destination = destination
        self.deadline = (
            deadline  # we treat this as deadline to leave the origin they arrive at
        )
        self.vot = vot

    def make_decision(self, price_lists):
        """
        we do a normalised probability over the reduced normalised price lists

        price_list is in the form (e, c, r)
        e is economic cost
        c is travel time
        r is road object
        """
        choice = choices(price_lists, weights=[x[1] for x in price_lists])
        m = max([x[1] for x in price_lists])
        # print([(m - x[1] + 1 / m) for x in price_lists], choice)
        return choice
        # TODO: I am fairly certain this is prioritising the option which is slowest....
        # double check the maths here

    def quantal_nominator(self, x, lambd=0.5):
        # print(x)
        """
        In the QRE model, individuals have a certain probability of choosing each available option, and this probability
        is influenced by the perceived benefits and costs of each option, as well as the behavior of others in the
        population. The parameter lambda represents the extent to which individuals take into account the behavior of
        others and adjust their choices accordingly.

        A higher value of lambda indicates that individuals are more rational in their decision-making process, meaning that
        they are more likely to choose the option that provides the highest expected payoff, based on their perceived
        benefits and costs. In this case, the influence of the behavior of others on their choices is relatively small.

        :param x:
        :param lambd:
        :return:
        """
        # if x > 709:
        #     x = 709
        try:
            return math.exp((lambd * x))
        except OverflowError:
            print(x, int(lambd * x))
            raise OverflowError

    def quantalify(self, r, rest, lambd=0.5):
        return np.exp(lambd * r) / np.sum(np.exp(lambd * r), axis=-0)

    def make_quantal_decision(self, route_costs):
        utility = [u[3] for u in route_costs]
        # # print(utility)
        # exponential_util = []
        # for x in utility:
        #     try:
        #         exponential_util.append(self.quantal_nominator(x))
        #     except OverflowError:
        #         exponential_util.append(float("inf"))
        # print(exponential_util)
        # quantal_weights = [sum(exponential_util) / u for u in exponential_util]
        # # print([(r1, r2) for (r1, r2, _, _) in route_costs], utility, exponential_util, quantal_weights)
        quantal_weights = [
            self.quantalify(x, np.asarray(utility, dtype=np.float32)) for x in utility
        ]
        choice = choices(route_costs, weights=quantal_weights)
        return choice

    def __repr__(self):
        return (
            "{Car: "
            + ", ".join(
                list(
                    map(
                        str,
                        [
                            self.k,
                            self.x0,
                            self.origin,
                            self.arrival,
                            self.destination,
                            self.deadline,
                            self.vot,
                        ],
                    )
                )
            )
            + "}"
        )
