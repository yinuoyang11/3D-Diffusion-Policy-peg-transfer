# DP3 dVRK Deploy Inference

`deploy_dp3_dvrk.py` provides a small Python API for loading one or more trained
DP3 checkpoints and running action-chunk inference from real dVRK observations.

## Inputs

Each `predict()` call expects:

- `task_name`: task alias from the registry.
- `point_cloud_world`: raw world-frame point cloud with shape `(N, 3)` or
  `(N, >=3)`.
- `agent_pos`: dVRK proprio vector with shape `(8,)`.
- `goal`: optional `(6,)` vector, required only for goal-conditioned checkpoints.

The deploy helper keeps the last `n_obs_steps` observations per task. On the
first call after `reset()`, it pads history by duplicating the current frame.

## Task Registry

Create a JSON file like:

```json
{
  "block": {
    "checkpoint_path": "outputs/dp3_peg_transfer_final/checkpoints/latest.ckpt"
  },
  "block_goal": {
    "checkpoint_path": "outputs/dp3_peg_transfer_goal/checkpoints/latest.ckpt",
    "num_points": 1024,
    "ground_z_min": 0.002,
    "bbox": [[-1.0, 1.0], [-1.0, 1.0], [0.002, 1.0]]
  }
}
```

`bbox` is optional. If omitted, the script keeps all non-ground finite points
before FPS downsampling.

## Python Usage

```python
from deploy_dp3_dvrk import DP3DVRKDeploy

deploy = DP3DVRKDeploy("tasks.json", device="cuda:0")

result = deploy.predict(
    "block_goal",
    point_cloud_world=pc_world,
    agent_pos=agent_pos,
    goal=goal,
)

action_chunk = result["action"]
meta = result["meta"]
```

`action_chunk` has shape `(n_action_steps, 8)`. The external robot controller is
responsible for deciding how many actions to execute before requesting a new
chunk.

## CLI Smoke Test

Prepare an `.npz` file with arrays named `point_cloud_world`, `agent_pos`, and
optionally `goal`, then run:

```bash
python deploy_dp3_dvrk.py --registry tasks.json --task block_goal --input sample_obs.npz --device cuda:0
```

Use `--lazy-load` to delay loading checkpoints until the first request for each
task.
