#!/usr/bin/env python3
"""Deploy helper for running trained DP3 policies on dVRK observations.

The main entry point is DP3DVRKDeploy. It accepts raw world-frame point clouds
and current robot proprioception, maintains the DP3 observation history, and
returns a full action chunk from the selected task policy.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


ArrayLike = Any


@dataclass
class TaskRuntime:
    alias: str
    checkpoint_path: pathlib.Path
    name: str | None = None
    num_points: int = 1024
    ground_z_min: float = 0.002
    bbox: Any = None
    workspace: Any = None
    policy: torch.nn.Module | None = None
    cfg: Any = None
    n_obs_steps: int | None = None
    expects_goal: bool = False
    history: dict[str, deque] = field(default_factory=dict)


class DP3DVRKDeploy:
    """Multi-task DP3 inference engine for dVRK deployment.

    Args:
        task_configs: Mapping from task alias to config dict, or path to a
            JSON/YAML registry. Each config needs "checkpoint_path" and may
            define "name", "num_points", "ground_z_min", and "bbox".
        device: Torch device used for policy inference.
        lazy_load: If False, load all checkpoints during initialization.
    """

    def __init__(
        self,
        task_configs: Mapping[str, Any] | str | pathlib.Path,
        device: str = "cuda:0",
        lazy_load: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.lazy_load = bool(lazy_load)
        self.tasks = self._parse_task_configs(task_configs)
        if not self.tasks:
            raise ValueError("task_configs must contain at least one task.")

        if not self.lazy_load:
            for alias in self.tasks:
                self._load_task(alias)

    def reset(self, task_name: str | None = None) -> None:
        """Clear observation history for one task, or all tasks."""
        if task_name is None:
            for task in self.tasks.values():
                self._reset_task_history(task)
            return
        task = self._get_task(task_name)
        self._reset_task_history(task)

    def predict(
        self,
        task_name: str,
        point_cloud_world: ArrayLike,
        agent_pos: ArrayLike,
        goal: ArrayLike | None = None,
    ) -> dict[str, Any]:
        """Run one DP3 inference call and return a full action chunk."""
        task = self._load_task(task_name)
        assert task.policy is not None
        assert task.n_obs_steps is not None

        processed_pc, pc_meta = self._preprocess_point_cloud(
            point_cloud_world,
            num_points=task.num_points,
            ground_z_min=task.ground_z_min,
            bbox=task.bbox,
        )
        proprio = self._as_vector(agent_pos, 8, "agent_pos")

        if task.expects_goal:
            if goal is None:
                raise ValueError(f"Task {task.alias!r} expects goal with shape (6,), but goal was not provided.")
            goal_vec = self._as_vector(goal, 6, "goal")
        else:
            goal_vec = None

        padded_history = self._append_history(task, processed_pc, proprio, goal_vec)
        obs_dict_np = self._build_obs_dict(task)
        obs_dict = {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device).unsqueeze(0)
            for key, value in obs_dict_np.items()
        }

        with torch.inference_mode():
            result = task.policy.predict_action(obs_dict)

        action = result["action"].detach().cpu().numpy()[0]
        action_pred = result["action_pred"].detach().cpu().numpy()[0]
        meta = {
            "task_name": task.alias,
            "model_name": task.name,
            "checkpoint_path": str(task.checkpoint_path),
            "device": str(self.device),
            "n_obs_steps": task.n_obs_steps,
            "history_padded": padded_history,
            **pc_meta,
        }
        return {
            "action": action,
            "action_pred": action_pred,
            "meta": meta,
        }

    @classmethod
    def _parse_task_configs(cls, task_configs: Mapping[str, Any] | str | pathlib.Path) -> dict[str, TaskRuntime]:
        if isinstance(task_configs, (str, pathlib.Path)):
            task_configs = load_task_registry(task_configs)
        if not isinstance(task_configs, Mapping):
            raise TypeError("task_configs must be a mapping or a JSON/YAML path.")

        tasks: dict[str, TaskRuntime] = {}
        for alias, cfg in task_configs.items():
            if isinstance(cfg, (str, pathlib.Path)):
                cfg = {"checkpoint_path": str(cfg)}
            if not isinstance(cfg, Mapping):
                raise TypeError(f"Task config for {alias!r} must be a mapping or checkpoint path.")
            if "checkpoint_path" not in cfg:
                raise ValueError(f"Task config for {alias!r} is missing required 'checkpoint_path'.")

            checkpoint_path = pathlib.Path(str(cfg["checkpoint_path"])).expanduser()
            if not checkpoint_path.is_absolute():
                checkpoint_path = pathlib.Path.cwd() / checkpoint_path
            tasks[str(alias)] = TaskRuntime(
                alias=str(alias),
                checkpoint_path=checkpoint_path,
                name=None if cfg.get("name") is None else str(cfg.get("name")),
                num_points=int(cfg.get("num_points", 1024)),
                ground_z_min=float(cfg.get("ground_z_min", 0.002)),
                bbox=cfg.get("bbox"),
            )
        return tasks

    def _get_task(self, task_name: str) -> TaskRuntime:
        try:
            return self.tasks[task_name]
        except KeyError as exc:
            known = ", ".join(sorted(self.tasks))
            raise KeyError(f"Unknown task {task_name!r}. Known tasks: {known}") from exc

    def _load_task(self, task_name: str) -> TaskRuntime:
        task = self._get_task(task_name)
        if task.policy is not None:
            return task
        if not task.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint for task {task.alias!r} not found: {task.checkpoint_path}")

        import dill
        from train import TrainDP3Workspace

        payload = torch.load(task.checkpoint_path.open("rb"), pickle_module=dill, map_location="cpu")
        workspace = TrainDP3Workspace(payload["cfg"])
        workspace.load_payload(payload, exclude_keys=("optimizer",))
        del payload

        use_ema = bool(getattr(workspace.cfg.training, "use_ema", False))
        policy = workspace.ema_model if use_ema and workspace.ema_model is not None else workspace.model
        policy.to(self.device)
        policy.eval()

        task.workspace = workspace
        task.policy = policy
        task.cfg = workspace.cfg
        task.n_obs_steps = int(workspace.cfg.n_obs_steps)
        task.expects_goal = "goal" in workspace.cfg.shape_meta.obs
        if task.name is None:
            task.name = str(getattr(workspace.cfg, "task_name", task.alias))
        self._reset_task_history(task)
        return task

    @staticmethod
    def _reset_task_history(task: TaskRuntime) -> None:
        maxlen = task.n_obs_steps if task.n_obs_steps is not None else 1
        task.history = {
            "point_cloud": deque(maxlen=maxlen),
            "agent_pos": deque(maxlen=maxlen),
            "goal": deque(maxlen=maxlen),
        }

    @classmethod
    def _preprocess_point_cloud(
        cls,
        point_cloud_world: ArrayLike,
        num_points: int,
        ground_z_min: float,
        bbox: Any = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        points = np.asarray(point_cloud_world, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"point_cloud_world must have shape (N, >=3), got {points.shape}")

        raw_count = int(points.shape[0])
        points = points[:, :3]
        finite_mask = np.all(np.isfinite(points), axis=1)
        points = points[finite_mask]
        finite_count = int(points.shape[0])

        points = points[points[:, 2] > ground_z_min]
        ground_filtered_count = int(points.shape[0])

        if bbox is not None:
            bounds = cls._parse_bbox(bbox)
            keep = (
                (points[:, 0] >= bounds[0, 0]) & (points[:, 0] <= bounds[0, 1]) &
                (points[:, 1] >= bounds[1, 0]) & (points[:, 1] <= bounds[1, 1]) &
                (points[:, 2] >= bounds[2, 0]) & (points[:, 2] <= bounds[2, 1])
            )
            points = points[keep]

        filtered_count = int(points.shape[0])
        if filtered_count == 0:
            raise ValueError("Point cloud is empty after finite, ground, and bbox filtering.")

        sampled = farthest_point_sampling(points, min(num_points, filtered_count))
        padded = False
        if sampled.shape[0] < num_points:
            pad_count = num_points - sampled.shape[0]
            repeats = np.resize(np.arange(sampled.shape[0], dtype=np.int64), pad_count)
            sampled = np.concatenate([sampled, sampled[repeats]], axis=0)
            padded = True

        meta = {
            "point_count_raw": raw_count,
            "point_count_finite": finite_count,
            "point_count_after_ground": ground_filtered_count,
            "point_count_filtered": filtered_count,
            "point_count_output": int(sampled.shape[0]),
            "point_cloud_padded": padded,
        }
        return sampled.astype(np.float32, copy=False), meta

    @staticmethod
    def _parse_bbox(bbox: Any) -> np.ndarray:
        if isinstance(bbox, Mapping):
            try:
                bbox = [bbox["x"], bbox["y"], bbox["z"]]
            except KeyError as exc:
                raise ValueError("bbox mapping must contain x, y, and z ranges.") from exc
        bounds = np.asarray(bbox, dtype=np.float32)
        if bounds.shape != (3, 2):
            raise ValueError(f"bbox must be [[xmin,xmax],[ymin,ymax],[zmin,zmax]], got {bounds.shape}")
        if np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError(f"bbox min values must be <= max values, got {bounds}")
        return bounds

    @staticmethod
    def _as_vector(value: ArrayLike, dim: int, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (dim,):
            raise ValueError(f"{name} must have shape ({dim},), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or Inf.")
        return arr

    def _append_history(
        self,
        task: TaskRuntime,
        point_cloud: np.ndarray,
        agent_pos: np.ndarray,
        goal: np.ndarray | None,
    ) -> bool:
        assert task.n_obs_steps is not None
        history = task.history
        history["point_cloud"].append(point_cloud)
        history["agent_pos"].append(agent_pos)
        if task.expects_goal:
            assert goal is not None
            history["goal"].append(goal)

        padded = False
        for key in ("point_cloud", "agent_pos", "goal"):
            if key == "goal" and not task.expects_goal:
                continue
            while len(history[key]) < task.n_obs_steps:
                history[key].appendleft(history[key][0].copy())
                padded = True
        return padded

    @staticmethod
    def _build_obs_dict(task: TaskRuntime) -> dict[str, np.ndarray]:
        obs = {
            "point_cloud": np.stack(list(task.history["point_cloud"]), axis=0).astype(np.float32),
            "agent_pos": np.stack(list(task.history["agent_pos"]), axis=0).astype(np.float32),
        }
        if task.expects_goal:
            obs["goal"] = np.stack(list(task.history["goal"]), axis=0).astype(np.float32)
        return obs


def farthest_point_sampling(
    points: np.ndarray,
    num_samples: int,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Farthest point sampling over xyz columns with deterministic first point."""
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected points shape (N, >=3), got {points.shape}")
    n_points = int(points.shape[0])
    if n_points == 0:
        raise ValueError("Input point cloud is empty.")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if n_points <= num_samples:
        indices = np.arange(n_points, dtype=np.int64)
        return (points, indices) if return_indices else points

    xyz = points[:, :3].astype(np.float32, copy=False)
    selected = np.empty(num_samples, dtype=np.int64)
    distances = np.full(n_points, np.inf, dtype=np.float32)
    farthest = 0
    for i in range(num_samples):
        selected[i] = farthest
        centroid = xyz[farthest]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = int(np.argmax(distances))

    sampled = points[selected]
    return (sampled, selected) if return_indices else sampled


def load_task_registry(path: str | pathlib.Path) -> dict[str, Any]:
    path = pathlib.Path(path)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML task registries.") from exc
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported registry suffix {path.suffix!r}; use .json, .yaml, or .yml.")


def _load_npz_input(path: str | pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        required = {"point_cloud_world", "agent_pos"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Input npz is missing required arrays: {sorted(missing)}")
        result = {
            "point_cloud_world": data["point_cloud_world"],
            "agent_pos": data["agent_pos"],
        }
        if "goal" in data.files:
            result["goal"] = data["goal"]
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test DP3 dVRK deploy inference.")
    parser.add_argument("--registry", required=True, help="JSON/YAML task registry.")
    parser.add_argument("--task", required=True, help="Task alias from the registry.")
    parser.add_argument("--input", required=True, help=".npz with point_cloud_world, agent_pos, optional goal.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lazy-load", action="store_true")
    args = parser.parse_args()

    deploy = DP3DVRKDeploy(args.registry, device=args.device, lazy_load=args.lazy_load)
    sample = _load_npz_input(args.input)
    result = deploy.predict(args.task, **sample)
    action = result["action"]
    print(f"action shape: {action.shape}")
    print(f"action_pred shape: {result['action_pred'].shape}")
    print(f"first action: {np.array2string(action[0], precision=6, suppress_small=True)}")
    print(f"meta: {json.dumps(_json_safe(result['meta']), indent=2)}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    main()
