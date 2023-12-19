from Models.DQN import DQNWrapper
from Models.Fixed import FixedAgentWrapper
from Models.Random import RandomAgentWrapper
from Models.Free import FreeAgentWrapper

agent_configs = {
    "DQN": {"agent": DQNWrapper},
    "Random": {"agent": RandomAgentWrapper},
    "Fixed": {"agent": FixedAgentWrapper},
    "Free": {"agent": FreeAgentWrapper},
}
