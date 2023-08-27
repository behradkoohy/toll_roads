import argparse
import os
import sqlite3
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
    parser.add_argument("--i_folder", help="The folder containing the exp_X directories")
    parser.add_argument("--output", help="The file to save the outputs to")
    args = parser.parse_args()

    # Get the list of exp_X directories
    exp_dirs = [
        folder for folder in os.listdir(args.i_folder)
        if os.path.isdir(os.path.join(args.i_folder, folder)) and folder.startswith("exp_")
    ]
    time_step_difference = []
    # For each exp_X directory, open the logging.db file
    for exp_dir in exp_dirs:
        print(os.path.join(os.path.join(args.i_folder, exp_dir), "logging.db" ))
        conn = open_logging_db(os.path.join(args.i_folder, exp_dir))
        cur = conn.cursor()
        cur.execute("SELECT MAX(EPOCH) FROM eval;")
        m_epoch = cur.fetchone()[0]
        cur.execute("SELECT TS_IN, TS_OUT FROM eval WHERE EPOCH=?;", (m_epoch, ))
        dbout = cur.fetchall()

        for row in dbout:
            time_step_difference.append(((row[1] - row[0]), row[0], row[1]))

        # Close the connection to the database
        conn.close()

    with open(args.output, 'w') as f:
        json.dump(time_step_difference, f, indent=4)


if __name__ == "__main__":
    main()
