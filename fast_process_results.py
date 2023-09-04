import argparse
import os
import sqlite3
import numpy as np
import json


def open_logging_db(exp_dir):
    """Opens the logging.db file in the exp_dir directory."""
    logging_db_path = os.path.join(exp_dir, "logging.db")
    conn = sqlite3.connect(logging_db_path)
    return conn

def main():
    """The main function."""
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--i_folder", help="The folder containing the exp_X directories"
    )
    parser.add_argument("--output", help="The file to save the outputs to")
    args = parser.parse_args()

    exp_dirs = [
        folder
        for folder in os.listdir(args.i_folder)
        if os.path.isdir(os.path.join(args.i_folder, folder))
           and folder.startswith("exp_")
    ]

    maxes = []
    mins = []
    avgs = []
    medians = []
    upqs = []
    lwqs = []

    for exp_dir in exp_dirs:
        print(os.path.join(os.path.join(args.i_folder, exp_dir), "logging.db"))
        conn = open_logging_db(os.path.join(args.i_folder, exp_dir))
        cur = conn.cursor()
        cur.execute("SELECT MAX(EPOCH) FROM eval;")
        m_epoch = cur.fetchone()[0]

        cur.execute("SELECT MAX(TS_OUT - TS_IN) FROM eval WHERE EPOCH=?;", (m_epoch,))
        maxes.append( cur.fetchone()[0])
        cur.execute("SELECT MIN(TS_OUT - TS_IN) FROM eval WHERE EPOCH=?;", (m_epoch,))
        mins.append( cur.fetchone()[0])
        cur.execute("SELECT AVG(TS_OUT - TS_IN) FROM eval WHERE EPOCH=?;", (m_epoch,))
        avgs.append( cur.fetchone()[0])

        cur.execute("SELECT (TS_OUT - TS_IN) FROM eval WHERE EPOCH=?;", (m_epoch,))
        ns = cur.fetchall()
        medians.append(np.median(ns))
        upqs.append(np.percentile(ns, 25))
        lwqs.append(np.percentile(ns, 75))

    output = {
        "maxs": np.average(maxes),
        "mins": np.average(mins),
        "avgs": np.average(avgs),
        "meds": np.average(medians),
        "upqs": np.average(upqs),
        "lwqs": np.average(lwqs),
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=4)




