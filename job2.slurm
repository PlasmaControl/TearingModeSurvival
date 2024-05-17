#!/bin/bash
#SBATCH --job-name=survival_tm         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=/scratch/gpfs/ar0535/error_files/slurm_%j.out 
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2023.9
conda activate torch-env

python train_survival_study.py
