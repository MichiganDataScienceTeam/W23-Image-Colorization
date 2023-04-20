#!/bin/bash

#SBATCH --job-name=kaggle_500k
#SBATCH --account=mdatascience_team
#SBATCH --partition=standard
#SBATCH --time=00-18:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --mail-user=casperg@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=8g

python3 train_kaggle.py
