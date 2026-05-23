from typing import Dict
import copy

import numpy as np
import torch
import zarr

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer


class PegTransferDataset(BaseDataset):
    """
    DP3 dataset adapter for the peg transfer zarr exported from
    Transfer_Learning/action_policy.

    This adapter uses observation history the same way as the DP3 paper/code:
    - point_cloud := last n_obs_steps frames from pointcloud_hist
    - agent_pos   := last n_obs_steps frames from proprio_hist
    - action      := pre-windowed action sequence stored in data/actions
    """

    def __init__(
        self,
        zarr_path,
        horizon=16,
        n_obs_steps=2,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        camera="camera_1",
        cache_in_memory=True,
        task_name=None,
        use_start_end_goal=False,
    ):
        super().__init__()
        self.zarr_path = zarr_path
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        self.camera = camera
        self.cache_in_memory = cache_in_memory
        self.task_name = task_name
        self.use_start_end_goal = use_start_end_goal

        root = zarr.open(zarr_path, mode="r")
        data = root["data"]
        self._meta = root["meta"]

        if camera in data:
            camera_data = data[camera]
            pointcloud_hist = camera_data["pointcloud_hist"]
        else:
            camera_data = None
            pointcloud_hist = data["pointcloud_hist"]

        proprio_hist = data["proprio_hist"]
        actions = data["actions"] if "actions" in data else data["action"]
        start_end_points = None
        start_end_points_valid = None
        if self.use_start_end_goal:
            if camera_data is None:
                raise KeyError(
                    f"use_start_end_goal=True requires data/{camera}/start_end_points, "
                    f"but camera group {camera!r} was not found."
                )
            if "start_end_points" not in camera_data or "start_end_points_valid" not in camera_data:
                raise KeyError(
                    f"use_start_end_goal=True requires data/{camera}/start_end_points and "
                    f"data/{camera}/start_end_points_valid."
                )
            start_end_points = np.asarray(camera_data["start_end_points"], dtype=np.float32)
            start_end_points_valid = np.asarray(camera_data["start_end_points_valid"], dtype=bool)

        if cache_in_memory:
            pointcloud_hist = np.asarray(pointcloud_hist, dtype=np.float32)
            proprio_hist = np.asarray(proprio_hist, dtype=np.float32)
            actions = np.asarray(actions, dtype=np.float32)

        if pointcloud_hist.shape[1] < self.n_obs_steps:
            raise ValueError(
                f"pointcloud_hist has only {pointcloud_hist.shape[1]} frames, "
                f"but n_obs_steps={self.n_obs_steps}"
            )
        if proprio_hist.shape[1] < self.n_obs_steps:
            raise ValueError(
                f"proprio_hist has only {proprio_hist.shape[1]} frames, "
                f"but n_obs_steps={self.n_obs_steps}"
            )

        self.point_cloud = pointcloud_hist[:, -self.n_obs_steps :].astype(np.float32)
        self.agent_pos = proprio_hist[:, -self.n_obs_steps :].astype(np.float32)
        self.action = actions.astype(np.float32)
        self.goal = None
        self.goal_valid = None
        if self.use_start_end_goal:
            if start_end_points.shape[:2] != (len(self.action), 2) or start_end_points.shape[-1] != 3:
                raise ValueError(
                    "Expected start_end_points shape (N, 2, 3), got "
                    f"{start_end_points.shape}"
                )
            if start_end_points_valid.shape != (len(self.action), 2):
                raise ValueError(
                    "Expected start_end_points_valid shape (N, 2), got "
                    f"{start_end_points_valid.shape}"
                )
            goal = start_end_points.reshape(len(self.action), 6).astype(np.float32)
            self.goal = np.repeat(goal[:, None, :], self.n_obs_steps, axis=1).astype(np.float32)
            self.goal_valid = start_end_points_valid

        if self.action.ndim != 3:
            raise ValueError(f"Expected actions shape (N, H, A), got {self.action.shape}")
        if self.action.shape[1] != self.horizon:
            raise ValueError(
                f"Dataset action horizon {self.action.shape[1]} does not match configured horizon {self.horizon}"
            )

        episode_ids = self._build_sample_episode_ids()
        train_episode_mask, val_episode_mask = self._build_episode_split_masks(episode_ids)
        train_indices = np.nonzero(train_episode_mask[episode_ids])[0]
        val_indices = np.nonzero(val_episode_mask[episode_ids])[0]

        self.indices = train_indices
        self._val_indices = val_indices
        if self.use_start_end_goal:
            selected_indices = np.concatenate([self.indices, self._val_indices])
            invalid_indices = selected_indices[~np.all(self.goal_valid[selected_indices], axis=1)]
            if len(invalid_indices) > 0:
                preview = invalid_indices[:10].tolist()
                raise ValueError(
                    "Found invalid start_end_points_valid entries for selected samples. "
                    f"First invalid sample indices: {preview}"
                )

    def _build_sample_episode_ids(self):
        if "sample_frame_indices" not in self._meta or "episode_ends" not in self._meta:
            return np.zeros(len(self.action), dtype=np.int64)

        sample_frame_indices = np.asarray(self._meta["sample_frame_indices"], dtype=np.int64)
        episode_ends = np.asarray(self._meta["episode_ends"], dtype=np.int64)
        if len(sample_frame_indices) != len(self.action):
            raise ValueError(
                "meta/sample_frame_indices length does not match number of samples: "
                f"{len(sample_frame_indices)} vs {len(self.action)}"
            )
        return np.searchsorted(episode_ends, sample_frame_indices, side="right")

    def _build_episode_split_masks(self, episode_ids):
        n_episodes = int(episode_ids.max()) + 1 if len(episode_ids) > 0 else 1
        episode_indices = np.arange(n_episodes, dtype=np.int64)

        rng = np.random.default_rng(self.seed)
        if self.max_train_episodes is not None and self.max_train_episodes < n_episodes:
            selected_train = rng.choice(episode_indices, size=self.max_train_episodes, replace=False)
            train_mask = np.zeros(n_episodes, dtype=bool)
            train_mask[selected_train] = True
            val_mask = ~train_mask
            return train_mask, val_mask

        if self.val_ratio <= 0.0 or n_episodes <= 1:
            train_mask = np.ones(n_episodes, dtype=bool)
            val_mask = np.zeros(n_episodes, dtype=bool)
            return train_mask, val_mask

        n_val = max(1, int(round(n_episodes * self.val_ratio)))
        n_val = min(n_val, n_episodes - 1)
        val_episodes = rng.choice(episode_indices, size=n_val, replace=False)
        val_mask = np.zeros(n_episodes, dtype=bool)
        val_mask[val_episodes] = True
        train_mask = ~val_mask
        return train_mask, val_mask

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.indices = self._val_indices
        val_set._val_indices = self.indices
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.action[self.indices],
            "agent_pos": self.agent_pos[self.indices],
            "point_cloud": self.point_cloud[self.indices],
        }
        if self.use_start_end_goal:
            data["goal"] = self.goal[self.indices]
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.action[self.indices])

    def __len__(self) -> int:
        return len(self.indices)

    def _sample_to_data(self, sample_idx):
        point_cloud = self.point_cloud[sample_idx].astype(np.float32)
        agent_pos = self.agent_pos[sample_idx].astype(np.float32)
        action = self.action[sample_idx].astype(np.float32)

        obs = {
            "point_cloud": point_cloud,
            "agent_pos": agent_pos,
        }
        if self.use_start_end_goal:
            obs["goal"] = self.goal[sample_idx].astype(np.float32)

        data = {
            "obs": obs,
            "action": action,
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx = int(self.indices[idx])
        data = self._sample_to_data(sample_idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
