#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=bk2g18@soton.ac.uk
#SBATCH --time=22:00:00
#SBATCH --nodes=1
#SBATCH --partition=amd

source tollroads/bin/activate
python Simulation.py -C $1 -T $2 -E $3 -L $4 -A $5 -CP $6

