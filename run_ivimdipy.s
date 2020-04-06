#!/bin/bash
#SBATCH --partition=cpu_long
#SBATCH --nodes=4
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --job-name=ivimproc_cluster
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=faye.mckenna@nyulangone.org
#SBATCH --time=3:14:15
#SBATCH --mem=100gb


module purge
module load python/cpu/3.6.5

cd /gpfs/home/fm1545/Faye/Diff_models/dmipy-master/
python run_ivim.py

echo "running python run_ivim.py"





