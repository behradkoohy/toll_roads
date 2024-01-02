from Simulation import Simulation

class Benchmark():
    def __init__(self):
        pass

    """
    We need to make this class run the simulation with the different 'manifests' (read: the order of which
    we run the agents). In between runs, we need to read the log file, create an output JSON/CSV (probably
    json) to store the results. Ideally, after we have ran it, we can output some graphs and nonsense that
    will show the performance of the agents.
    
    
    """
    def run(self):
        sim = Simulation()

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
