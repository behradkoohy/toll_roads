
class TravellingCar():
    def __init__(self, car, id, timeIn, currentETA, initialETA, road, vot):
        self.car = car
        self.id = id
        self.timeIn = timeIn
        self.currentETA = currentETA
        self.initialETA = initialETA
        self.road = road
        self.vot = vot

class RoadQueueManager():
    def __init__(self, simulation):
        self.simulation = simulation
        self.arrived_vehicles = 0
        self.clearQueue()
        # we need to do some stuff to calculate the reward
        self.clearRewards()

    def clearRewards(self):
        self.roadRewards = {r: 0.0 for r in self.simulation.toll_roads}

    def addToQueue(self, road, travellingCar: TravellingCar):

        self.roadQueues[road] = self.roadQueues[road] + [travellingCar]
        self.roadRewards[road] = self.roadRewards.get(road, 0.0) + self.simulation.road_cost[road]

    def updateQueue(self):
        current_time = self.simulation.current_timestep
        done_vehicles = []
        for road, queue in self.roadQueues.items():
            current_road_travel_time = road.get_road_travel_time()
            new_queue = []
            for car in queue:
                # check if car has arrived
                # case 1: current time is the ETA in which case it has arrived on time
                # case 2: new ETA is less than the current time, in which case it arrives early
                if car.currentETA == current_time or car.timeIn + current_road_travel_time < current_time:
                    self.arrived_vehicles += 1
                    done_vehicles.append([
                        car.id,
                        self.simulation.epoch,
                        car.timeIn,
                        self.simulation.current_timestep,
                        str(hash(road)),
                        car.vot
                    ])
                    continue
                elif current_time + current_road_travel_time < car.currentETA:
                    car.currentETA = current_time + current_road_travel_time
                    new_queue.append(car)
                else:
                    new_queue.append(car)
            # set new queue
            self.roadQueues[road] = new_queue
            # update the travel time for the road
            road.t_curr = len(new_queue)
            # print(self.arrived_vehicles)
        self.simulation.log.batch_add_new_completed_vehicle(done_vehicles)
        # self.roadRewards = {r: 0.0 for r in self.simulation.toll_roads}

    def updateTravelTime(self):
        self.simulation.get_road_time_cost()

    def clearQueue(self):
        self.arrived_vehicles = 0
        self.roadQueues = {}
        for road in self.simulation.get_roads():
            road.t_curr = 0
            self.roadQueues[road] = []

    def getQueueDemand(self):
        return [len(x) for x in self.roadQueues.values()]

    def getArrivedVehicles(self):
        return self.arrived_vehicles

    def getCarsOnRoad(self, road):
        return len(self.roadQueues[road])

    def getTimeStepsUntilTimeReduction(self, road):
        if not self.roadQueues[road]:
            return 0
        return self.roadQueues[road][0].currentETA - self.simulation.current_timestep