import sqlite3

import matplotlib.pyplot as plt
import seaborn as sns


class Logging:
    def __init__(self, db_file="logging.db", delete_table=True):
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        if delete_table:
            self.cursor.execute("DROP TABLE IF EXISTS results;")
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS results (
                EPOCH INTEGER NOT NULL,
                TIMESTEP INT NOT NULL,
                OBSERV_AGENT_1 BLOB NOT NULL,
                ACTION_AGENT_1 BLOB NOT NULL,
                REWARD_AGENT_1 BLOB NOT NULL,
                OBSERV_AGENT_2 BLOB NOT NULL,
                ACTION_AGENT_2 BLOB NOT NULL,
                REWARD_AGENT_2 BLOB NOT NULL,
                PRIMARY KEY (epoch, timestep)
            )"""
        )

    def add_timestep_results(self, epoch, timestep, obs1, rew1, act1, obs2, rew2, act2):
        self.cursor.execute(
            """INSERT INTO results 
        (EPOCH, TIMESTEP, OBSERV_AGENT_1, ACTION_AGENT_1, REWARD_AGENT_1, OBSERV_AGENT_2, ACTION_AGENT_2, REWARD_AGENT_2)
        VALUES 
        (?, ?, ?, ?, ?, ?, ?, ?)""",
            (epoch, timestep, obs1, rew1, act1, obs2, rew2, act2),
        )
        self.conn.commit()

    def removing_half_complete_run(self, epoch):
        self.cursor.execute("DELETE FROM results WHERE EPOCH=?", (epoch, ))
        self.conn.commit()

    def pretty_graphs(self):
        grapher = PrettyGraphs(self.conn)



class PrettyGraphs:
    def __init__(self, db_connection, max_timestep=None, max_epoch=None):
        self.conn = db_connection
        self.cursor = self.conn.cursor()

        if max_timestep is None:
            self.cursor.execute(
                """
                SELECT MIN(max_timestep)
                FROM (
                  SELECT MAX(TIMESTEP) AS max_timestep
                  FROM results
                  GROUP BY EPOCH
                );
            """
            )
            self.max_timestep = self.cursor.fetchall()[0][0]
        else:
            self.max_timestep = max_timestep

        if max_epoch is None:
            self.cursor.execute(
                """
                SELECT MAX(EPOCH) FROM results;
                """
            )
            self.max_epoch = self.cursor.fetchall()[0][0]
        else:
            self.max_epoch = max_epoch

        self.total_reward()

    def total_reward(self):
        ordered_agent_performance = []
        for x in range(self.max_epoch):
            self.cursor.execute(
                "SELECT REWARD_AGENT_1, REWARD_AGENT_2 FROM results WHERE TIMESTEP=? AND EPOCH=?",
                (self.max_timestep, x),
            )
            ordered_agent_performance.append(self.cursor.fetchall()[0])
        print(ordered_agent_performance)

        ordered_data = {
            'agent_1': [x[0] for x in ordered_agent_performance],
            'agent_2': [x[1] for x in ordered_agent_performance],
        }
        sns.lineplot(ordered_data)
        plt.show()
