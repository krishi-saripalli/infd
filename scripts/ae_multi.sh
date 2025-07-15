#!/bin/bash

#SBATCH -J infd_ae #name of the job
#SBATCH --nodes=1                          # number of nodes to reserve
#SBATCH -n 8                   # number of CPU cores
#SBATCH --mem=256gb                          # memory per node
#SBATCH -t 72:00:00                         
#SBATCH --partition=3090-gcondo --gres=gpu:8 # partition and number of GPUs (per node)
#SBATCH --export=CXX=g++                    # compiler
#SBATCH -o /users/ksaripal/logs/infd_ae.out   #out logs
#SBATCH -e /users/ksaripal/logs/infd_ae.err   #error logs


cd /users/ksaripal/BVC/infd/

module load anaconda/2023.09-0-7nso27y
source /gpfs/runtime/opt/anaconda/2023.03-1/etc/profile.d/conda.sh
conda activate infd

torchrun --standalone --nproc-per-node=8 run.py --cfg cfgs/ae_custom_h5.yaml --save-root /users/ksaripal/data/ksaripal/infd/ckpt --name ae_voronoi_cond_$(date +%Y%m%d_%H%M%S) -w