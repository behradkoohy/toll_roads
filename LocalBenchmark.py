import argparse
import os

import numpy as np
from tqdm import tqdm, trange
from Simulation import Simulation
import sqlite3


class ResultProcessor:
    def __init__(self, db_file="results.db", delete_table=True, logdir=""):
        self.conn_out = sqlite3.connect(db_file)
        self.cur_out = self.conn_out.cursor()
        self.logdir = logdir
        if delete_table:
            self.cur_out.execute("DROP TABLE IF EXISTS metaresult_tratime;")
        self.cur_out.execute(
            """CREATE TABLE IF NOT EXISTS metaresult_tratime (
                AGENT TEXT NOT NULL,
                REPEAT INTEGER NOT NULL,
                MIN REAL NOT NULL,
                Q1 REAL NOT NULL,
                MEDIAN REAL NOT NULL,
                MEAN REAL NOT NULL,
                Q3 REAL NOT NULL,
                MAX REAL NOT NULL,
                PRIMARY KEY (AGENT, REPEAT)
            )
            """
        )
        if delete_table:
            self.cur_out.execute("DROP TABLE IF EXISTS metaresult_vottime;")
        self.cur_out.execute(
            """CREATE TABLE IF NOT EXISTS metaresult_vottime (
                AGENT TEXT NOT NULL,
                REPEAT INTEGER NOT NULL,
                MIN REAL NOT NULL,
                Q1 REAL NOT NULL,
                MEDIAN REAL NOT NULL,
                MEAN REAL NOT NULL,
                Q3 REAL NOT NULL,
                MAX REAL NOT NULL,
                PRIMARY KEY (AGENT, REPEAT)
            )
            """
        )

        # THESE ARE ONLY FOR USE WITH THE ADAPTIVE AGENTS
        self.maxes = []
        self.mins = []
        self.avgs = []
        self.medians = []
        self.upqs = []
        self.lwqs = []

    def process_fixed_result(self, agent, db_path="logging.db"):
        self.conn_in = self.open_logging_db(db_path)
        self.cur_in = self.conn_in.cursor()

        self.cur_in.execute("SELECT MAX(EPOCH) FROM eval;")
        m_epoch = self.cur_in.fetchone()[0]
        for x in trange(m_epoch + 1, unit="timesteps"):
            self.cur_in.execute(
                "SELECT MAX((TS_OUT - TS_IN)*VEH_VOT) FROM eval WHERE EPOCH=?;", (x,)
            )
            maxs = self.cur_in.fetchone()[0]

            self.cur_in.execute(
                "SELECT MIN((TS_OUT - TS_IN)*VEH_VOT) FROM eval WHERE EPOCH=?;", (x,)
            )
            mins = self.cur_in.fetchone()[0]

            self.cur_in.execute(
                "SELECT AVG((TS_OUT - TS_IN)*VEH_VOT) FROM eval WHERE EPOCH=?;", (x,)
            )
            avgs = self.cur_in.fetchone()[0]

            self.cur_in.execute(
                "SELECT ((TS_OUT - TS_IN)*VEH_VOT) FROM eval WHERE EPOCH=?;", (x,)
            )
            ns = self.cur_in.fetchall()
            medians = np.median(ns)
            upqs = np.percentile(ns, 25)
            lwqs = np.percentile(ns, 75)
            self.cur_out.execute(
                "INSERT INTO metaresult_vottime (AGENT, REPEAT, MIN, Q1, MEDIAN, MEAN, Q3, MAX) VALUES (?,?,?,?,?,?,?,?)",
                (agent, x, mins, lwqs, medians, avgs, upqs, maxs),
            )
            self.cur_in.execute(
                "SELECT MAX(TS_OUT - TS_IN) FROM eval WHERE EPOCH=?;", (x,)
            )
            maxs = self.cur_in.fetchone()[0]

            self.cur_in.execute(
                "SELECT MIN(TS_OUT - TS_IN) FROM eval WHERE EPOCH=?;", (x,)
            )
            mins = self.cur_in.fetchone()[0]

            self.cur_in.execute(
                "SELECT AVG(TS_OUT - TS_IN) FROM eval WHERE EPOCH=?;", (x,)
            )
            avgs = self.cur_in.fetchone()[0]

            self.cur_in.execute(
                "SELECT (TS_OUT - TS_IN) FROM eval WHERE EPOCH=?;", (x,)
            )
            ns = self.cur_in.fetchall()
            medians = np.median(ns)
            upqs = np.percentile(ns, 25)
            lwqs = np.percentile(ns, 75)

            self.cur_out.execute(
                "INSERT INTO metaresult_tratime (AGENT, REPEAT, MIN, Q1, MEDIAN, MEAN, Q3, MAX) VALUES (?,?,?,?,?,?,?,?)",
                (agent, x, mins, lwqs, medians, avgs, upqs, maxs),
            )
            self.conn_out.commit()

    def process_adapt_result(self, agent, repeat_id, db_path="logging.db"):
        self.conn_in = self.open_logging_db(db_path)
        self.cur_in = self.conn_in.cursor()

        self.cur_in.execute("SELECT MAX(EPOCH) FROM eval;")
        m_epoch = self.cur_in.fetchone()[0]

        self.cur_in.execute(
            "SELECT MAX((TS_OUT - TS_IN)*VEH_VOT) FROM eval WHERE EPOCH=?;", (m_epoch,)
        )
        maxs = self.cur_in.fetchone()[0]

        self.cur_in.execute(
            "SELECT MIN((TS_OUT - TS_IN)*VEH_VOT) FROM eval WHERE EPOCH=?;", (m_epoch,)
        )
        mins = self.cur_in.fetchone()[0]

        self.cur_in.execute(
            "SELECT AVG((TS_OUT - TS_IN)*VEH_VOT) FROM eval WHERE EPOCH=?;", (m_epoch,)
        )
        avgs = self.cur_in.fetchone()[0]

        self.cur_in.execute(
            "SELECT ((TS_OUT - TS_IN)*VEH_VOT) FROM eval WHERE EPOCH=?;", (m_epoch,)
        )
        ns = self.cur_in.fetchall()
        medians = np.median(ns)
        upqs = np.percentile(ns, 25)
        lwqs = np.percentile(ns, 75)
        self.cur_out.execute(
            "INSERT INTO metaresult_vottime (AGENT, REPEAT, MIN, Q1, MEDIAN, MEAN, Q3, MAX) VALUES (?,?,?,?,?,?,?,?)",
            (agent, repeat_id, mins, lwqs, medians, avgs, upqs, maxs),
        )
        self.cur_in.execute(
            "SELECT MAX(TS_OUT - TS_IN) FROM eval WHERE EPOCH=?;", (m_epoch,)
        )
        maxs = self.cur_in.fetchone()[0]

        self.cur_in.execute(
            "SELECT MIN(TS_OUT - TS_IN) FROM eval WHERE EPOCH=?;", (m_epoch,)
        )
        mins = self.cur_in.fetchone()[0]

        self.cur_in.execute(
            "SELECT AVG(TS_OUT - TS_IN) FROM eval WHERE EPOCH=?;", (m_epoch,)
        )
        avgs = self.cur_in.fetchone()[0]

        self.cur_in.execute(
            "SELECT (TS_OUT - TS_IN) FROM eval WHERE EPOCH=?;", (m_epoch,)
        )
        ns = self.cur_in.fetchall()
        medians = np.median(ns)
        upqs = np.percentile(ns, 25)
        lwqs = np.percentile(ns, 75)

        self.cur_out.execute(
            "INSERT INTO metaresult_tratime (AGENT, REPEAT, MIN, Q1, MEDIAN, MEAN, Q3, MAX) VALUES (?,?,?,?,?,?,?,?)",
            (agent, repeat_id, mins, lwqs, medians, avgs, upqs, maxs),
        )
        self.conn_out.commit()

    def open_logging_db(self, db_path):
        """Opens the logging.db file in the exp_dir directory."""
        # logging_db_path = db_path
        conn = sqlite3.connect(db_path)
        return conn


class Benchmark:
    def __init__(
        self, cars, timesteps, n_repeat, tollroads, freeroads, logdir, capacity
    ):
        self.n_cars = cars
        self.timesteps = timesteps
        self.n_repeat = n_repeat
        self.n_tollroads = tollroads
        self.n_freeroads = freeroads
        self.logdir = logdir
        self.capacity = capacity
        self.processor = ResultProcessor(db_file=self.logdir + os.sep + "results.db")

    """
    We need to make this class run the simulation with the different 'manifests' (read: the order of which
    we run the agents). In between runs, we need to read the log file, create an output JSON/CSV (probably
    json) to store the results. Ideally, after we have ran it, we can output some graphs and nonsense that
    will show the performance of the agents.

    1. First, we should run the 'fixed' benchmarks 
    2. Then we should evaluate the result of each one as soon as it is run
    
    How do we store this data? We will create a SQLite3 database and store the results in there.
    A simple database - the primary key can just be 'Agent' and then we keep the min, q1, median, average, q3, max
    
    lets cook (yum)
    """

    def run(self):
        fixed_agents = ["Random", "Fixed", "Free"]
        for agent in fixed_agents:
            print(1)
            sim = Simulation(
                self.n_cars,
                self.timesteps,
                agent=agent,
                n_toll_roads=self.n_tollroads,
                n_free_road=self.n_freeroads,
                log_dir=self.logdir,
                fixed_capacity=self.capacity,
            )
            self.processor.process_fixed_result(
                agent, db_path=self.logdir + os.sep + "logging.db"
            )

        for n in trange(self.n_repeat, position=0, leave=False):
            sim = Simulation(
                self.n_cars,
                self.timesteps,
                agent="DQN",
                n_epochs=10,
                n_toll_roads=self.n_tollroads,
                n_free_road=self.n_freeroads,
                log_dir=self.logdir,
                fixed_capacity=self.capacity,
            )
            self.processor.process_adapt_result(
                "DQN", n, db_path=self.logdir + os.sep + "logging.db"
            )

    def capacity_experiment_runs(self, delete_table=True):
        # self.conn_out = sqlite3.connect(self.logdir + os.sep + "results.db")
        # self.cur_out = self.conn_out.cursor()
        self.conn_out = self.processor.conn_out
        self.cur_out = self.processor.cur_out
        if delete_table:
            self.cur_out.execute("DROP TABLE IF EXISTS capacity_runs_tratime;")
            self.cur_out.execute("DROP TABLE IF EXISTS capacity_runs_vottime;")
        self.cur_out.execute(
            """CREATE TABLE IF NOT EXISTS capacity_runs_tratime (
                AGENT TEXT NOT NULL,
                CAPACITY INTEGER NOT NULL,
                MIN REAL NOT NULL,
                Q1 REAL NOT NULL,
                MEDIAN REAL NOT NULL,
                MEAN REAL NOT NULL,
                Q3 REAL NOT NULL,
                MAX REAL NOT NULL,
                PRIMARY KEY (AGENT, CAPACITY)
            )
            """
        )
        self.cur_out.execute(
            """CREATE TABLE IF NOT EXISTS capacity_runs_vottime (
                AGENT TEXT NOT NULL,
                CAPACITY INTEGER NOT NULL,
                MIN REAL NOT NULL,
                Q1 REAL NOT NULL,
                MEDIAN REAL NOT NULL,
                MEAN REAL NOT NULL,
                Q3 REAL NOT NULL,
                MAX REAL NOT NULL,
                PRIMARY KEY (AGENT, CAPACITY)
            )
            """
        )
        for capacity in range(1,401):
            print("CAPACITY", capacity, "------------------------------------")
            self.cur_out.execute(
                """CREATE TABLE IF NOT EXISTS metaresult_vottime (
                    AGENT TEXT NOT NULL,
                    REPEAT INTEGER NOT NULL,
                    MIN REAL NOT NULL,
                    Q1 REAL NOT NULL,
                    MEDIAN REAL NOT NULL,
                    MEAN REAL NOT NULL,
                    Q3 REAL NOT NULL,
                    MAX REAL NOT NULL,
                    PRIMARY KEY (AGENT, REPEAT)
                )
                """
            )
            self.cur_out.execute(
                """CREATE TABLE IF NOT EXISTS metaresult_tratime (
                    AGENT TEXT NOT NULL,
                    REPEAT INTEGER NOT NULL,
                    MIN REAL NOT NULL,
                    Q1 REAL NOT NULL,
                    MEDIAN REAL NOT NULL,
                    MEAN REAL NOT NULL,
                    Q3 REAL NOT NULL,
                    MAX REAL NOT NULL,
                    PRIMARY KEY (AGENT, REPEAT)
                )
                """
            )
            fixed_agents = ["Random", "Fixed", "Free"]
            for agent in fixed_agents:
                sim = Simulation(
                    self.n_cars,
                    self.timesteps,
                    n_epochs=5,
                    agent=agent,
                    n_toll_roads=self.n_tollroads,
                    n_free_road=self.n_freeroads,
                    log_dir=self.logdir,
                    fixed_capacity=capacity,
                )

                self.processor.process_fixed_result(
                    agent, db_path=self.logdir + os.sep + "logging.db"
                )
                self.cur_out.execute(
                    """SELECT AGENT, MIN, Q1, MEDIAN, MEAN, Q3, MAX FROM metaresult_vottime;
                    """
                )
                vots = self.cur_out.fetchall()
                mins = np.mean([x[1] for x in vots])
                lwqs = np.mean([x[5] for x in vots])
                medians = np.mean([x[3] for x in vots])
                avgs = np.mean([x[4] for x in vots])
                upqs = np.mean([x[2] for x in vots])
                maxs = np.mean([x[6] for x in vots])
                self.cur_out.execute(
                    """
                INSERT INTO capacity_runs_vottime (AGENT, CAPACITY, MIN, Q1, MEDIAN, MEAN, Q3, MAX) VALUES (?,?,?,?,?,?,?,?)
                """,
                    ((agent, capacity, mins, lwqs, medians, avgs, upqs, maxs)),
                )
                self.cur_out.execute(
                    """SELECT AGENT, MIN, Q1, MEDIAN, MEAN, Q3, MAX FROM metaresult_tratime;
                    """
                )
                vots = self.cur_out.fetchall()
                mins = np.mean([x[1] for x in vots])
                lwqs = np.mean([x[5] for x in vots])
                medians = np.mean([x[3] for x in vots])
                avgs = np.mean([x[4] for x in vots])
                upqs = np.mean([x[2] for x in vots])
                maxs = np.mean([x[6] for x in vots])
                self.cur_out.execute(
                    """
                INSERT INTO capacity_runs_tratime (AGENT, CAPACITY, MIN, Q1, MEDIAN, MEAN, Q3, MAX) VALUES (?,?,?,?,?,?,?,?)
                """,
                    ((agent, capacity, mins, lwqs, medians, avgs, upqs, maxs)),
                )
            self.cur_out.execute("DROP TABLE IF EXISTS metaresult_tratime;")
            self.cur_out.execute("DROP TABLE IF EXISTS metaresult_vottime;")
            self.conn_out.commit()



if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="MMRP Simulation")
    ap.add_argument("-C", "--cars", default=300, action="store", type=int)
    ap.add_argument("-T", "--timesteps", default=200, action="store", type=int)
    ap.add_argument("-E", "--epochs", default=100, action="store", type=int)
    ap.add_argument("-TR", "--tollroads", default=2, action="store", type=int)
    ap.add_argument("-FR", "--freeroads", default=0, action="store", type=int)
    ap.add_argument("-L", "--logdir", default="./", action="store", type=str)
    # ap.add_argument(
    #     "-A",
    #     "--agent",
    #     default="Random",
    #     choices=[agent for agent in agent_configs],
    #     type=str,
    # )
    ap.add_argument("-CP", "--capacity", default=100, action="store", type=int)
    a = ap.parse_args()
    b = Benchmark(
        a.cars, a.timesteps, 1, a.tollroads, a.freeroads, a.logdir, a.capacity
    )
    b.capacity_experiment_runs()
