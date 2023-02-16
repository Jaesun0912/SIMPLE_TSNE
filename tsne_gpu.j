#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # Cores per node
#SBATCH --partition=gpu    # Partition name (skylake)
##
#SBATCH --job-name="tsne_gpu"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

### To choose GPU nodes, turn on the option below...
# export CUDA_VISIBLE_DEVICES= 0

stdbuf -oL python run.py --tsne > tsne.log
