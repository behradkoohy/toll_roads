import codecs
import pickle
import sqlite3
import json

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
                ACTION_AGENT_1 INT NOT NULL,
                REWARD_AGENT_1 REAL NOT NULL,
                PRICE_AGENT_1 REAL NOT NULL,
                OBSERV_AGENT_2 BLOB NOT NULL,
                ACTION_AGENT_2 INT NOT NULL,
                REWARD_AGENT_2 REAL NOT NULL,
                PRICE_AGENT_2 REAL NOT NULL,
                PRIMARY KEY (epoch, timestep)
            );"""
        )
        if delete_table:
            self.cursor.execute("DROP TABLE IF EXISTS eval;")
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS eval (
                ID INTEGER NOT NULL,
                EPOCH INTEGER NOT NULL,
                TS_IN INTEGER NOT NULL,
                TS_OUT INTEGER NOT NULL,
                ROUTE INTEGER NOT NULL,
                VEH_VOT REAL NOT NULL,
                PRIMARY KEY (ID, EPOCH)
            );"""
        )

    def set_titles(self, titles):
        self.titles = titles

    def add_timestep_results(
        self, epoch, timestep, obs1, act1, rew1, pri1, obs2, act2, rew2, pri2
    ):
        self.cursor.execute(
            """INSERT INTO results 
        (EPOCH, TIMESTEP, OBSERV_AGENT_1, ACTION_AGENT_1, REWARD_AGENT_1, PRICE_AGENT_1, OBSERV_AGENT_2, ACTION_AGENT_2, REWARD_AGENT_2, PRICE_AGENT_2)
        VALUES 
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            (
                epoch,
                timestep,
                obs1,
                int(act1),
                float(rew1),
                float(pri1),
                obs2,
                int(act2),
                float(rew2),
                float(pri2),
            ),
        )

    def add_new_completed_vehicle(self, id, epoch, ts_in, ts_out, route, veh_vot):
        self.cursor.execute(
            """ INSERT INTO eval
            (ID, EPOCH, TS_IN, TS_OUT, ROUTE, VEH_VOT)
            VALUES 
            (?,?,?,?,?,?)
            """,
            (id, epoch, ts_in, ts_out, route, veh_vot),
        )

    def removing_half_complete_run(self, epoch):
        self.cursor.execute("DELETE FROM results WHERE EPOCH=?", (epoch,))
        self.conn.commit()

    def pretty_graphs(self):
        grapher = PrettyGraphs(self.conn, self.titles)

    def main(self):
        pass


class PrettyGraphs:
    def __init__(self, db_connection, titles, max_timestep=None, max_epoch=None):
        self.conn = db_connection
        self.cursor = self.conn.cursor()
        self.titles = titles
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

        # self.total_reward()
        self.five_timeline_plots()

    def read_obs_blob(self, s):
        if not isinstance(s, str):
            print(s)
        return pickle.loads(codecs.decode(s.encode(), "base64"))

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
            "agent_1": [x[0] for x in ordered_agent_performance],
            "agent_2": [x[1] for x in ordered_agent_performance],
        }
        if None in ordered_data["agent_1"]:
            breakpoint()
        if None in ordered_data["agent_2"]:
            breakpoint()

        sns.lineplot(ordered_data)
        plt.show()

    def five_timeline_plots(self):
        # epochs = sorted([random.randint(0, self.max_epoch) for _ in range(5)]) + [self.max_epoch]
        # epochs = [
        #     1,
        #     int(0.2 * self.max_epoch),
        #     int(0.4 * self.max_epoch),
        #     int(0.6 * self.max_epoch),
        #     int(0.8 * self.max_epoch),
        #     self.max_epoch,
        # ]
        epochs = [
            # 1,
            int(0.33 * self.max_epoch),
            int(0.66 * self.max_epoch),
            self.max_epoch,
        ]

        for e in epochs:
            fig, ax = plt.subplots(nrows=3, ncols=1)
            fig.tight_layout(pad=10)
            self.cursor.execute(
                """SELECT TIMESTEP, 
                REWARD_AGENT_1, 
                REWARD_AGENT_2,
                ACTION_AGENT_1, 
                ACTION_AGENT_2, 
                PRICE_AGENT_1, 
                PRICE_AGENT_2 
                FROM results WHERE EPOCH=? ORDER BY TIMESTEP
                """,
                (e,),
            )
            db_out = self.cursor.fetchall()
            ordered_data_rew = {
                # 'x': [x[0] for x in db_out],
                "agent_1": [x[1] for x in db_out],
                "agent_2": [x[2] for x in db_out],
            }
            print(
                "agent_1 Cum_rew",
                sum([x[1] for x in db_out]),
                "agent_2 Cum_rew",
                sum([x[2] for x in db_out]),
                "total:",
                sum([x[1] for x in db_out] + [x[2] for x in db_out]),
            )
            ordered_data_act = {
                # 'x': [x[0] for x in db_out],
                "agent_1": [x[3] for x in db_out],
                "agent_2": [x[4] for x in db_out],
            }
            ordered_data_pri = {
                # 'x': [x[0] for x in db_out],
                "agent_1": [x[5] for x in db_out],
                "agent_2": [x[6] for x in db_out],
            }
            sns.lineplot(ordered_data_rew, ax=ax[0])
            ax[0].set_ylabel("REWARD: epoch " + str(e))
            sns.lineplot(ordered_data_act, ax=ax[1])
            ax[1].set_ylabel("ACTION: epoch " + str(e))
            ax[1].set_ylim([-1.5, 1.5])
            sns.lineplot(ordered_data_pri, ax=ax[2])
            ax[2].set_ylabel("PRICE: epoch " + str(e))
            plt.autoscale()
            plt.show()

            self.cursor.execute(
                """SELECT TIMESTEP, 
                OBSERV_AGENT_1, 
                OBSERV_AGENT_2
                FROM results WHERE EPOCH=? ORDER BY TIMESTEP
                """,
                (e,),
            )
            db_out = self.cursor.fetchall()
            # breakpoint()
            # for agent in db_out:
            #     array
            # print(self.read_obs_blob(db_out[e][1]), self.read_obs_blob(db_out[e][2]))

            ordered_data_obs = {
                "agent_1": [self.read_obs_blob(x[1]) for x in db_out],
                "agent_2": [self.read_obs_blob(x[2]) for x in db_out],
            }
            fig, ax = plt.subplots(
                nrows=len(ordered_data_obs["agent_1"][0]), ncols=1, figsize=[7, 40]
            )
            # for a1, a2 in zip(ordered_data_obs['agent_1'], ordered_data_obs['agent_2']):
            # print(ordered_data_obs['agent_1'])
            # titles = self.tit
            for i in range(len(ordered_data_obs["agent_1"][0])):
                sns.lineplot(
                    {
                        "a1": [x[i] for x in ordered_data_obs["agent_1"]],
                        "a2": [x[i] for x in ordered_data_obs["agent_2"]],
                    },
                    ax=ax[i],
                )
                ax[i].title.set_text(self.titles[i])
            fig.tight_layout(pad=5.0)
            plt.autoscale()
            plt.show()


class ManifestMaker:
    def __init__(self, manifest_dir):
        self.manifest_dir = manifest_dir

    def __write_manifest(self, contents, filepath):
        with open(filepath, "w") as f:
            json.dump(contents, f, indent=4)

    def __create_empty_manifest(self, filepath):
        with open(filepath, "w") as f:
            json.dump([], f, indent=4)

    def __append_manifest(self, contents, filepath):
        file_data = json.load(open(filepath))
        with open(filepath, "w") as f:
            json.dump(file_data+[contents], f, indent=4)

    def create_model_manifest(self):
        self.__create_empty_manifest(self.manifest_dir + "/model_manifest.json")

    def write_model_manifest(self, data):
        self.__append_manifest(data, self.manifest_dir + "/model_manifest.json")

    def write_simulation_manifest(self, data):
        self.__write_manifest(data, self.manifest_dir + "/simulation_manifest.json")

    # def write_environment_manifest(self, data):
    #     self.__write_manifest(data, self.manifest_dir + "/simulation_manifest.json")
