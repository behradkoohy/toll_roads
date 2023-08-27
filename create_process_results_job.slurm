#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=bk2g18@soton.ac.uk
#SBATCH --time=22:00:00
#SBATCH --nodes=1
#SBATCH --partition=amd

source tollroads/bin/activate
python process_results.py --i_folder $1 --output $2
