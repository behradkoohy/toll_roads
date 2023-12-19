import math


def volume_delay_function(a, b, c, t0, v):
    """
    :param v: volume of cars on road currently
    :param a: VDF calibration parameter alpha, values of 0.15 default
    :param b: VDF calibration parameter beta, value of 4 default
    :param c: road capacity
    :param t0: free flow travel time
    :return: travel time for a vehicle on road

    link for default value info:
    https://www.degruyter.com/document/doi/10.1515/eng-2022-0022/html?lang=en
    """
    #a = 0.15
    a = 0.656
    b = 4.8
    #b = 4
    # c = 30
    # t0 = 30

    return t0 * (1 + (a * pow((v / c), b)))


def agent_probability_function(x, k, x0):
    """
    :param x: variable input, cost of road
    :param k: APF calibration parameter k, steepness control. must be negative or zero (stable probability)
    :param x0: APF calibration parameter x0, transition midpoint parameter
    :return: real value between 0,1 representing choice probability
    """
    if k < 0:
        raise Exception(
            "K must be more than 0, as otherwise probability of agent taking road increases with price increase"
        )

    return 1.0 / (1 + math.exp((k) * (x0 - x)))
