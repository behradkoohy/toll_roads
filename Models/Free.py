import random

class FreeAgentWrapper:
    def __init__(self, *args):
        self.model = FreeAgent()
        # return self

    # def act(self, obs):
    #     return self.model.act()

    def observe(self, *args):
        pass

class FreeAgent:
    def __init__(self):
        self.cost = 0

    def act(self, act):
        return self.cost

    def observe(self):
        pass



