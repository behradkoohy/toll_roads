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
    parser.add_argument("--folder", help="The folder containing the exp_X directories")
    args = parser.parse_args()

    # Get the list of exp_X directories
    exp_dirs = [
        folder for folder in os.listdir(args.folder)
        if os.path.isdir(os.path.join(args.folder, folder)) and folder.startswith("exp_")
    ]

    time_step_difference = []
    # For each exp_X directory, open the logging.db file
    for exp_dir in exp_dirs:
        conn = open_logging_db(exp_dir)
        cur = conn.cursor()
        cur.execute("SELECT TS_IN, TS_OUT FROM eval")
        dbout = cur.fetchall()

        for row in dbout:
            time_step_difference.append(row[1] - row[0])

        # Close the connection to the database
        conn.close()

    with open('exp_results_out.json', 'w') as f:
        json.dump(time_step_difference, f, indent=4)


if __name__ == "__main__":
    main()
