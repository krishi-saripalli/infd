#!/bin/bash
#SBATCH --job-name=latent_voronoi
#SBATCH --output=/users/ksaripal/logs/latent_voronoi.log
#SBATCH --error=/users/ksaripal/logs/latent_voronoi.err
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --mem=256g
#SBATCH -t 02-00:00:00
#SBATCH -p 3090-gcondo --gres=gpu:8
#SBATCH --export=CXX=g++

cd /users/ksaripal/BVC/infd/


module load anaconda/2023.09-0-7nso27y
source /gpfs/runtime/opt/anaconda/2023.03-1/etc/profile.d/conda.sh
conda activate infd


accelerate launch --config_file OneNoise/accelerate.yml \
    OneNoise/train.py \
    --model_config medium \
    --latent_diffusion \
    --latent_dataset_path /users/ksaripal/data/ksaripal/infd/latent/ae_voronoi_cond_20250515_002907/ \
    --ae_config_path cfgs/ae_custom_h5.yaml \
    --ae_checkpoint_path /users/ksaripal/data/ksaripal/infd/ckpt/ae_voronoi_cond_20250515_002907/last-model.pth \
    --substance_params_dir /users/ksaripal/data/ksaripal/infd/image \
    --noise_types voronoi \
    --out_dir /users/ksaripal/data/ksaripal/infd/out/ \
    --exp_name latent_voronoi_medium_test \
    --batch_size 128 \
    --grad_accum 2 \
    --sample_every 1000 \
    --train_num_steps 200000 \
    --lr 8e-5 \
    --precision fp32 \
    --auto_normalize_latents False \
    --z_score_latents True
