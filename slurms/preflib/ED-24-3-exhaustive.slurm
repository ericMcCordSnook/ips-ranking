#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH -p standard
#SBATCH --output=job_output/out
#SBATCH --error=job_output/error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=esm7ky@virginia.edu
load module anaconda2
python run_experiment.py config/experiments/preflib/ED-24-3-exhaustive.yml
