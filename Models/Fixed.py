import random

class FixedAgentWrapper:
    def __init__(self, *args):
        self.model = FixedAgent()
        # return self

    # def act(self, obs):
    #     return self.model.act()

    def observe(self, *args):
        pass

class FixedAgent:
    def __init__(self):
        self.cost = 30

    def act(self, act):
        return self.cost

    def observe(self):
        pass



