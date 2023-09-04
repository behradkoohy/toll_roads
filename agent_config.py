from Models.DQN import DQNWrapper
from Models.Fixed import FixedAgentWrapper
from Models.Random import RandomAgentWrapper

agent_configs = {
    "DQN": {"agent": DQNWrapper},
    "Random": {"agent": RandomAgentWrapper},
    "Fixed": {"agent": FixedAgentWrapper},
}
