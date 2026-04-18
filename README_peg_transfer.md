# DP3 Training on `episode_all_cameras1.zarr`

This repository contains a small custom adaptation of the original
`3D-Diffusion-Policy` codebase so it can train DP3 on the peg transfer dataset
collected in the `Transfer_Learning` project.

## What was added

The peg-transfer-specific changes are mainly in:

- `3D-Diffusion-Policy/diffusion_policy_3d/dataset/peg_transfer_dataset.py`
- `3D-Diffusion-Policy/diffusion_policy_3d/config/task/peg_transfer.yaml`

In addition, the outer repository contains:

- `Dockerfile`
- `requirements_train.txt`

These are the server-side files used to build a training container.

## Dataset layout expected by DP3

The dataset adapter expects a zarr file with the same structure as:

`Transfer_Learning/task/peg_transfer/data/peg_task1/data/episode_all_cameras1.zarr`

The fields used by `PegTransferDataset` are:

- `data/<camera_name>/pointcloud_hist`
- `data/proprio_hist`
- `data/actions` or `data/action`
- `meta/sample_frame_indices`
- `meta/episode_ends`

For example, when `camera=camera_1`, the point cloud input is read from:

`data/camera_1/pointcloud_hist`

## How the zarr data is converted into DP3 input

For each training sample:

- `obs.point_cloud` uses the last `n_obs_steps` frames from `pointcloud_hist`
- `obs.agent_pos` uses the last `n_obs_steps` frames from `proprio_hist`
- `action` uses the pre-windowed action chunk stored in `data/actions`

Concretely, in `peg_transfer_dataset.py`:

- `self.point_cloud = pointcloud_hist[:, -self.n_obs_steps :]`
- `self.agent_pos = proprio_hist[:, -self.n_obs_steps :]`
- `self.action = actions`

So if `n_obs_steps=2`, each training sample contains:

- 2 frames of point cloud history
- 2 frames of robot state history
- 1 action chunk of length `horizon`

## Task config used for peg transfer

The peg transfer task config is:

`3D-Diffusion-Policy/diffusion_policy_3d/config/task/peg_transfer.yaml`

Key settings:

- `obs.point_cloud.shape = [1024, 3]`
- `obs.agent_pos.shape = [8]`
- `action.shape = [8]`
- `dataset._target_ = diffusion_policy_3d.dataset.peg_transfer_dataset.PegTransferDataset`

By default the config is set to use:

- `camera: camera_1`
- `n_obs_steps: 2`

## Example training command

Run from:

`3D-Diffusion-Policy/3D-Diffusion-Policy`

Example:

```bash
python train.py --config-name=dp3.yaml task=peg_transfer n_obs_steps=2 task.dataset.camera=camera_1 task.dataset.zarr_path=/workspace/Transfer_Learning/task/peg_transfer/data/peg_task1/data/episode_all_cameras1.zarr training.device=cuda:0 logging.mode=offline hydra.run.dir=outputs/dp3_peg_transfer hydra.sweep.dir=outputs/dp3_peg_transfer
```

## Notes

- This adapter uses the existing DP3 training loop. No separate train script is
  needed.
- The zarr dataset is assumed to already contain windowed action targets.
- To train a different single-view variant, only change
  `task.dataset.camera=camera_k`.
- To test multi-view generalization, train with one camera and evaluate with
  point clouds generated from other cameras at inference time.
