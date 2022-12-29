from functools import partial
from utils import agent_probability_function
from random import choices


class Car:
    def __init__(self, k, x0, arrival, deadline, origin, destination, budget):
        self.k = k
        self.x0 = x0
        self.determine_funct = partial(agent_probability_function, k, x0)
        self.origin = origin
        self.arrival = arrival
        self.destination = destination
        self.deadline = (
            deadline  # we treat this as deadline to leave the origin they arrive at
        )
        self.budget = budget

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
                            self.budget,
                        ],
                    )
                )
            )
            + "}"
        )
