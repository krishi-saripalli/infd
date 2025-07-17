CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 run.py --cfg cfgs/ae_custom_h5.yaml --save-root /users/ksaripal/data/ksaripal/infd/ckpt --name ae_voronoi

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 run.py --cfg cfgs/ae_custom_h5.yaml --save-root /users/ksaripal/data/ksaripal/infd/ckpt --name ae_voronoi


Voronoi checkpoint is at /users/ksaripal/data/ksaripal/infd/ckpt/ae_voronoi_cond_20250515_002907/last-model.pth

To create latent dataset with pre-trained encoder
```
python OneNoise/create_latents.py \
    --ae_train_config cfgs/ae_custom_h5.yaml \
    --ae_checkpoint_path /users/ksaripal/data/ksaripal/infd/ckpt/ae_voronoi_cond_20250515_002907/last-model.pth \
    --output_path /users/ksaripal/data/ksaripal/infd/latent/ae_voronoi_cond_20250515_002907/latent.hdf5
```



- TODO: We need to update `create_latents.py`. If the data was augmented with cutmix, then it's spatial paramters need to be saved. It's probably less tedious to just bake the creation/saving of the latents along with their spatial paramters into the checkpointing step of the AE experiment!

Once the latents have been made, you can run a quick debug to test to view stats and a sample reconstruction

```
python -m OneNoise.decode_latents \
    --latent_hdf5_path /users/ksaripal/data/ksaripal/infd/latent/ae_voronoi_cond_20250515_002907/latent.hdf5 \
    --ae_config_path cfgs/ae_custom_h5.yaml \
    --ae_checkpoint_path /users/ksaripal/data/ksaripal/infd/ckpt/ae_voronoi_cond_20250515_002907/last-model.pth \
    --noise_type voronoi \
    --sample_index 0 \
    --output_path ./decoded_images/test.png \
    --output_size 256 
 
    
```

> TODO: Fix `vertical_strip` outputs so that they correctly pass the correct gr4id dimensions to the INFD decoder + renderer.

To run a single GPU debug version of latent diffusion, run 

```
python OneNoise/train.py     --model_config medium     --latent_diffusion     --latent_dataset_path /users/ksaripal/data/ksaripal/infd/latent/ae_voronoi_20250416_002955/     --ae_config_path cfgs/ae_custom_h5.yaml     --ae_checkpoint_path /users/ksaripal/data/ksaripal/infd/ckpt/ae_voronoi_20250416_002955/last-model.pth     --noise_types voronoi     --out_dir /users/ksaripal/data/ksaripal/infd/out/     --exp_name latent_voronoi_medium_test     --batch_size 4     --grad_accum 2     --sample_every 1     --train_num_steps 50000     --lr 8e-5     --precision fp32 --auto_normalize_latents False  --z_score_latents True 

```

To test the model run

```
python OneNoise/test.py \
    --out_dir OneNoise/pretrained \
    --exp_name tiny_spherical \
    --checkpoint latent-voronoi-51.pt \
    --latent_diffusion \
    --ae_config_path cfgs/ae_custom_h5.yaml \
    --ae_checkpoint_path /users/ksaripal/data/ksaripal/infd/ckpt/ae_voronoi_20250416_002955/last-model.pth \
    --output_file tiny_spherical_voronoi_interpolation.png \
    --sample_timesteps 50 \
    --device cuda:0 \
    --model_config medium
```

The model seems to respond well to uniform parameters, but fails to take on the blending behaviour given per pixel spatial conditoning.
TODO: we need to conditon the *AE* in the same way as the diffusion model.