CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 run.py --cfg cfgs/ae_custom_h5.yaml --save-root /users/ksaripal/data/ksaripal/infd/ckpt --name ae_voronoi -w
