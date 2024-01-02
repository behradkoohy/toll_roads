import random

class RandomAgentWrapper:
    def __init__(self, *args):
        self.model = RandomAgent()
        # return self

    # def act(self, obs):
    #     return self.model.act()

    def observe(self, *args):
        pass

class RandomAgent:
    def __init__(self, *args):
        self.min = 0
        self.max = 2

    def act(self, act):
        return random.randint(self.min, self.max)

    def observe(self):
        pass



