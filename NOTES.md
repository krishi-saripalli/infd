CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 run.py --cfg cfgs/ae_custom_h5.yaml --save-root /users/ksaripal/data/ksaripal/infd/ckpt --name ae_voronoi

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 run.py --cfg cfgs/ae_custom_h5.yaml --save-root /users/ksaripal/data/ksaripal/infd/ckpt --name ae_voronoi

# Understanding Trainer Iteration Parameters in `ae.yaml`


*   `max_iter`:
    *   **Meaning:** The total number of training iterations (batches processed) the entire training process will run for. Training stops once `self.iter` reaches this value.

*   `epoch_iter`:
    *   **Meaning:** Defines the number of training iterations that constitute one "epoch" *for the purpose of the outer training loop and logging*. It doesn't necessarily correspond to a full pass over the dataset unless the dataset size and batch size align perfectly. The main loop iterates `max_iter // epoch_iter` times.

*   `eval_iter`:
    *   **Meaning:** Specifies how many training iterations (`self.iter`) should pass before the validation evaluation (`self.evaluate()`) is triggered.
    *   **Calculation:** The evaluation happens when `epoch % (eval_iter // epoch_iter) == 0`. Setting `eval_iter` equal to `epoch_iter` means evaluation occurs after every `epoch_iter` training iterations. Setting `eval_iter` low relative to `epoch_iter` (like `eval_iter=10, epoch_iter=1000`) causes frequent evaluations, significantly slowing down the overall training progress in terms of iterations per second, as evaluation takes time but doesn't increment `self.iter`.

*   `save_iter`:
    *   **Meaning:** Specifies how many training iterations (`self.iter`) should pass before an intermediate checkpoint (e.g., `iter-XXX.pth`) is saved.
    *   **Calculation:** Saving happens when `epoch % (save_iter // epoch_iter) == 0`. `last-model.pth` and potentially `best-model.pth` are saved more frequently (usually every epoch).

*   `vis_iter`:
    *   **Meaning:** Specifies how many training iterations (`self.iter`) should pass before the visualization function (`self.visualize()`) is triggered.
    *   **Calculation:** Visualization happens when `epoch % (vis_iter // epoch_iter) == 0`.

**Example:**

If `max_iter = 10000`, `epoch_iter = 1000`, `eval_iter = 2000`, `vis_iter = 5000`:
*   Training runs for 10,000 iterations total.
*   The outer loop runs for `10000 / 1000 = 10` epochs.
*   Evaluation runs every `2000 / 1000 = 2` epochs (at iterations 2000, 4000, 6000, 8000, 10000).
*   Visualization runs every `5000 / 1000 = 5` epochs (at iterations 5000, 10000).

The code in `trainers/base_trainer.py` (around lines 249-261) enforces the following divisibility rules via assertions:

*   `max_iter` **must** be divisible by `epoch_iter`.
*   If `save_iter` is specified (i.e., not `null`), it **must** be divisible by `epoch_iter`.
*   If `eval_iter` is specified (i.e., not `null`), it **must** be divisible by `epoch_iter`.
*   If `vis_iter` is specified (i.e., not `null`), it **must** be divisible by `epoch_iter`.

These constraints ensure that evaluation, saving, and visualization events align correctly with the calculated epoch boundaries used in the main training loop.


Voronoi checkpoint is at /users/ksaripal/data/ksaripal/infd/ckpt/ae_voronoi_20250416_002955/last-model.pth

To create latent dataset with pre-trained encoder
```
python OneNoise/create_latents.py \
    --ae_train_config cfgs/ae_custom_h5.yaml \
    --ae_checkpoint_path /users/ksaripal/data/ksaripal/infd/ckpt/ae_voronoi_20250416_002955/last-model.pth \
    --output_path /users/ksaripal/data/ksaripal/infd/latent/ae_voronoi_20250416_002955/latent.hdf5
```



- We need to update `create_latents.py`. If the data was augmented with cutmix, then it's spatial paramters need to be saved. It's probably less tedious to just bake the creation/saving of the latents along with their spatial paramters into the checkpointing step of the AE experiment!