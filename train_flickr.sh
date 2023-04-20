#!/bin/bash

#SBATCH --job-name=flickr_300k
#SBATCH --account=mdatascience_team
#SBATCH --partition=standard
#SBATCH --time=02-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --mail-user=casperg@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=16g

python3 train_flickr.py
