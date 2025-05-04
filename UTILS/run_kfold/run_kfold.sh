#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=pbatch
#SBATCH --account=ncov2019
#SBATCH --time=720
#SBATCH --export=ALL
#SBATCH --job-name="run_kfold_GP_RF"

#SBATCH --output="/g/g16/apaulson/workspace/gitlab_repos/ucsf-atom-nek/featurize_data_slurm/feat_add_valid.out"
#SBATCH --error="/g/g16/apaulson/workspace/gitlab_repos/ucsf-atom-nek/featurize_data_slurm/feat_add_valid.out"

date
cd /g/g16/apaulson/workspace/gitlab_repos/ucsf-atom-nek/featurize_data_slurm/

python GP_kfold10x.py ./featurized/
python RF_kfold10x.py ./featurized/