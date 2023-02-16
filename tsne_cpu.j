#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24         # Cores per node
#SBATCH --partition=loki3    # Partition name (skylake)
##
#SBATCH --job-name="tsne_cpu"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

OMP_NUM_THREADS=$SLURM_NTASKS stdbuf -oL python run.py --tsne > tsne.log
