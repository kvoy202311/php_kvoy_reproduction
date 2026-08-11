from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


_REQUIRED_MOTION_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def _motion_file_signature(path: Path) -> str:
    """Return a path-independent identity for one motion file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"{path.name}:{digest.hexdigest()}"


def _read_optional_names(data: np.lib.npyio.NpzFile, key: str, expected_count: int, motion_file: Path):
    if key not in data.files:
        return None

    names_array = np.asarray(data[key])
    if names_array.ndim != 1 or names_array.shape[0] != expected_count:
        raise ValueError(
            f"Invalid '{key}' in {motion_file}: expected shape ({expected_count},), got {names_array.shape}."
        )
    return tuple(str(name) for name in names_array.tolist())


class MotionLoader:
    """Load and validate one whole-body tracking motion NPZ."""

    def __init__(self, motion_file: str | Path, body_indexes: Sequence[int], device: str = "cpu"):
        self.motion_file = Path(motion_file).expanduser().resolve()
        if not self.motion_file.is_file():
            raise FileNotFoundError(f"Motion file does not exist or is not a file: {self.motion_file}")
        if self.motion_file.suffix.lower() != ".npz":
            raise ValueError(f"Motion file must have a .npz suffix: {self.motion_file}")

        with np.load(self.motion_file, allow_pickle=False) as data:
            missing_keys = [key for key in _REQUIRED_MOTION_KEYS if key not in data.files]
            if missing_keys:
                raise KeyError(f"Motion file {self.motion_file} is missing required arrays: {missing_keys}")

            fps_array = np.asarray(data["fps"])
            if fps_array.size != 1:
                raise ValueError(f"Invalid 'fps' in {self.motion_file}: expected one scalar, got {fps_array.shape}.")
            self.fps = float(fps_array.reshape(-1)[0])
            if not np.isfinite(self.fps) or self.fps <= 0.0:
                raise ValueError(
                    f"Invalid 'fps' in {self.motion_file}: expected a positive finite value, got {self.fps}."
                )

            arrays = {key: np.asarray(data[key]) for key in _REQUIRED_MOTION_KEYS if key != "fps"}
            self._validate_arrays(arrays)

            self.joint_names = _read_optional_names(data, "joint_names", arrays["joint_pos"].shape[1], self.motion_file)
            self.body_names = _read_optional_names(data, "body_names", arrays["body_pos_w"].shape[1], self.motion_file)

        self._body_indexes = torch.as_tensor(body_indexes, dtype=torch.long, device=device)
        if self._body_indexes.ndim != 1 or self._body_indexes.numel() == 0:
            raise ValueError("body_indexes must be a non-empty one-dimensional sequence.")
        if torch.any(self._body_indexes < 0) or torch.any(self._body_indexes >= arrays["body_pos_w"].shape[1]):
            raise IndexError(
                f"Tracked body index is outside the motion body range [0, {arrays['body_pos_w'].shape[1] - 1}] "
                f"for {self.motion_file}."
            )

        self._joint_pos = torch.as_tensor(arrays["joint_pos"], dtype=torch.float32, device=device)
        self._joint_vel = torch.as_tensor(arrays["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.as_tensor(arrays["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.as_tensor(arrays["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.as_tensor(arrays["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.as_tensor(arrays["body_ang_vel_w"], dtype=torch.float32, device=device)

        self.time_step_total = self._joint_pos.shape[0]
        self.joint_count = self._joint_pos.shape[1]
        self.body_count = self._body_pos_w.shape[1]
        self.num_motions = 1
        self.motion_lengths = torch.tensor([self.time_step_total], dtype=torch.long, device=device)
        self.motion_start_idx = torch.zeros(1, dtype=torch.long, device=device)
        self.motion_end_idx = self.motion_lengths.clone()
        self.motion_files = (str(self.motion_file),)
        self.motion_signatures = (_motion_file_signature(self.motion_file),)

    def _validate_arrays(self, arrays: dict[str, np.ndarray]):
        joint_pos = arrays["joint_pos"]
        joint_vel = arrays["joint_vel"]
        if joint_pos.ndim != 2:
            raise ValueError(f"Invalid 'joint_pos' in {self.motion_file}: expected [T, J], got {joint_pos.shape}.")
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(
                f"Invalid 'joint_vel' in {self.motion_file}: expected {joint_pos.shape}, got {joint_vel.shape}."
            )

        time_step_total = joint_pos.shape[0]
        if time_step_total < 2:
            raise ValueError(f"Motion {self.motion_file} must contain at least two frames, got {time_step_total}.")

        expected_body_shapes = {
            "body_pos_w": (time_step_total, None, 3),
            "body_quat_w": (time_step_total, None, 4),
            "body_lin_vel_w": (time_step_total, None, 3),
            "body_ang_vel_w": (time_step_total, None, 3),
        }
        body_count = arrays["body_pos_w"].shape[1] if arrays["body_pos_w"].ndim == 3 else None
        for key, (expected_time, _, expected_width) in expected_body_shapes.items():
            array = arrays[key]
            expected_shape = (expected_time, body_count, expected_width)
            if array.ndim != 3 or array.shape != expected_shape:
                raise ValueError(
                    f"Invalid '{key}' in {self.motion_file}: expected {expected_shape}, got {array.shape}."
                )

        for key, array in arrays.items():
            if not np.issubdtype(array.dtype, np.number):
                raise TypeError(f"Invalid '{key}' in {self.motion_file}: expected numeric data, got {array.dtype}.")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Invalid '{key}' in {self.motion_file}: data contains NaN or infinity.")

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MultiMotionLoader:
    """Load all NPZ files in one directory as separate clips in one motion dataset."""

    def __init__(self, motion_dir: str | Path, body_indexes: Sequence[int], device: str = "cpu"):
        self.motion_dir = Path(motion_dir).expanduser().resolve()
        if not self.motion_dir.is_dir():
            raise NotADirectoryError(f"Motion directory does not exist or is not a directory: {self.motion_dir}")

        motion_files = sorted(self.motion_dir.glob("*.npz"))
        if not motion_files:
            raise FileNotFoundError(f"No .npz motion files found directly inside: {self.motion_dir}")

        loaders = [MotionLoader(path, body_indexes, device=device) for path in motion_files]
        reference = loaders[0]
        for loader in loaders[1:]:
            self._validate_compatible(reference, loader)

        self.fps = reference.fps
        self.joint_names = reference.joint_names
        self.body_names = reference.body_names
        self.joint_count = reference.joint_count
        self.body_count = reference.body_count
        self._body_indexes = reference._body_indexes
        self.motion_files = tuple(str(path.resolve()) for path in motion_files)
        self.motion_signatures = tuple(loader.motion_signatures[0] for loader in loaders)

        self.motion_lengths = torch.tensor(
            [loader.time_step_total for loader in loaders], dtype=torch.long, device=device
        )
        self.motion_end_idx = self.motion_lengths.cumsum(dim=0)
        self.motion_start_idx = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), self.motion_end_idx[:-1]], dim=0
        )
        self.num_motions = len(loaders)

        self._joint_pos = torch.cat([loader._joint_pos for loader in loaders], dim=0)
        self._joint_vel = torch.cat([loader._joint_vel for loader in loaders], dim=0)
        self._body_pos_w = torch.cat([loader._body_pos_w for loader in loaders], dim=0)
        self._body_quat_w = torch.cat([loader._body_quat_w for loader in loaders], dim=0)
        self._body_lin_vel_w = torch.cat([loader._body_lin_vel_w for loader in loaders], dim=0)
        self._body_ang_vel_w = torch.cat([loader._body_ang_vel_w for loader in loaders], dim=0)
        self.time_step_total = self._joint_pos.shape[0]

    @staticmethod
    def _validate_compatible(reference: MotionLoader, candidate: MotionLoader):
        if not np.isclose(reference.fps, candidate.fps, rtol=0.0, atol=1.0e-6):
            raise ValueError(
                f"Motion FPS mismatch: {reference.motion_file} uses {reference.fps:g} Hz, "
                f"but {candidate.motion_file} uses {candidate.fps:g} Hz."
            )

        tensor_names = (
            "_joint_pos",
            "_joint_vel",
            "_body_pos_w",
            "_body_quat_w",
            "_body_lin_vel_w",
            "_body_ang_vel_w",
        )
        for tensor_name in tensor_names:
            reference_shape = getattr(reference, tensor_name).shape[1:]
            candidate_shape = getattr(candidate, tensor_name).shape[1:]
            if reference_shape != candidate_shape:
                raise ValueError(
                    f"Motion shape mismatch for {tensor_name.removeprefix('_')}: {reference.motion_file} has "
                    f"{reference_shape}, but {candidate.motion_file} has {candidate_shape}."
                )

        for names_key in ("joint_names", "body_names"):
            if getattr(reference, names_key) != getattr(candidate, names_key):
                raise ValueError(
                    f"Motion {names_key} mismatch between {reference.motion_file} and {candidate.motion_file}."
                )

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


def load_motion_dataset(
    *,
    motion_file: str | Path | None,
    motion_dir: str | Path | None,
    body_indexes: Sequence[int],
    device: str = "cpu",
) -> MotionLoader | MultiMotionLoader:
    """Load exactly one configured motion source."""

    if (motion_file is None) == (motion_dir is None):
        raise ValueError("Configure exactly one of 'motion_file' or 'motion_dir'.")
    if motion_dir is not None:
        return MultiMotionLoader(motion_dir, body_indexes, device=device)
    return MotionLoader(motion_file, body_indexes, device=device)


def advance_motion_frames(
    motion_ids: torch.Tensor,
    time_steps: torch.Tensor,
    motion_end_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance one frame without ever crossing a concatenated clip boundary.

    The returned frame is clamped to the current clip's final valid frame.  The
    boolean result marks environments that have just completed that final
    frame and therefore need either a new clip or an episode termination.
    """

    if motion_ids.shape != time_steps.shape:
        raise ValueError(
            f"motion_ids and time_steps must have equal shapes, got {motion_ids.shape} and {time_steps.shape}."
        )
    if motion_ids.dtype != torch.long or time_steps.dtype != torch.long:
        raise TypeError("motion_ids and time_steps must use torch.long indices.")
    if motion_end_idx.ndim != 1 or motion_end_idx.numel() == 0:
        raise ValueError("motion_end_idx must be a non-empty one-dimensional tensor.")
    if torch.any(motion_ids < 0) or torch.any(motion_ids >= motion_end_idx.numel()):
        raise IndexError("motion_ids contains an index outside motion_end_idx.")

    final_frames = motion_end_idx[motion_ids] - 1
    next_frames = torch.minimum(time_steps + 1, final_frames)
    motion_completed = next_frames >= final_frames
    return next_frames, motion_completed


def advance_motion_frames_with_final_hold(
    motion_ids: torch.Tensor,
    time_steps: torch.Tensor,
    motion_end_idx: torch.Tensor,
    final_hold_counts: torch.Tensor,
    final_hold_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance a motion while holding its final frame for a minimum number of policy steps.

    ``final_hold_steps=0`` preserves the original clip-boundary behavior.  A
    positive value clamps the reference to the clip's final frame and counts
    complete policy steps spent there before reporting completion.
    """

    if final_hold_counts.shape != time_steps.shape:
        raise ValueError(
            f"final_hold_counts and time_steps must have equal shapes, got {final_hold_counts.shape} and "
            f"{time_steps.shape}."
        )
    if final_hold_counts.dtype != torch.long:
        raise TypeError("final_hold_counts must use torch.long values.")
    if final_hold_steps < 0:
        raise ValueError(f"final_hold_steps must be non-negative, got {final_hold_steps}.")

    final_frames = motion_end_idx[motion_ids] - 1
    was_on_final_frame = time_steps >= final_frames
    next_frames, reached_final_frame = advance_motion_frames(motion_ids, time_steps, motion_end_idx)

    next_hold_counts = torch.where(
        was_on_final_frame,
        final_hold_counts + 1,
        torch.zeros_like(final_hold_counts),
    )
    if final_hold_steps == 0:
        motion_completed = reached_final_frame
    else:
        motion_completed = reached_final_frame & (next_hold_counts >= final_hold_steps)
    return next_frames, next_hold_counts, motion_completed


def deterministic_motion_starts(
    env_ids: torch.Tensor,
    motion_start_idx: torch.Tensor,
    *,
    mode: str,
    fixed_motion_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign deterministic clips and their exact frame-zero indices."""

    if env_ids.ndim != 1 or env_ids.dtype != torch.long:
        raise TypeError("env_ids must be a one-dimensional torch.long tensor.")
    if motion_start_idx.ndim != 1 or motion_start_idx.numel() == 0:
        raise ValueError("motion_start_idx must be a non-empty one-dimensional tensor.")

    num_motions = int(motion_start_idx.numel())
    if mode == "fixed":
        if fixed_motion_id < 0 or fixed_motion_id >= num_motions:
            raise ValueError(f"fixed_motion_id must be in [0, {num_motions - 1}], got {fixed_motion_id}.")
        motion_ids = torch.full_like(env_ids, fixed_motion_id)
    elif mode == "round_robin":
        motion_ids = torch.remainder(env_ids, num_motions)
    else:
        raise ValueError(f"Deterministic motion mode must be 'fixed' or 'round_robin', got {mode!r}.")
    return motion_ids, motion_start_idx[motion_ids]


def apply_forced_motion_starts(
    motion_ids: torch.Tensor,
    sampled_time_steps: torch.Tensor,
    motion_start_idx: torch.Tensor,
    force_start_mask: torch.Tensor,
) -> torch.Tensor:
    """Force selected environments to use their clip's first frame.

    This keeps phase restrictions separate from motion selection.  It is used
    by terrain-aware initialization: nominal-height environments may retain a
    sampled phase, while non-nominal environments start before any contact
    with the obstacle.
    """

    if motion_ids.shape != sampled_time_steps.shape or motion_ids.shape != force_start_mask.shape:
        raise ValueError(
            "motion_ids, sampled_time_steps, and force_start_mask must have equal shapes, "
            f"got {motion_ids.shape}, {sampled_time_steps.shape}, and {force_start_mask.shape}."
        )
    if motion_ids.dtype != torch.long or sampled_time_steps.dtype != torch.long:
        raise TypeError("motion_ids and sampled_time_steps must use torch.long indices.")
    if force_start_mask.dtype != torch.bool:
        raise TypeError("force_start_mask must use torch.bool values.")
    if motion_start_idx.ndim != 1 or motion_start_idx.numel() == 0:
        raise ValueError("motion_start_idx must be a non-empty one-dimensional tensor.")
    if torch.any(motion_ids < 0) or torch.any(motion_ids >= motion_start_idx.numel()):
        raise IndexError("motion_ids contains an index outside motion_start_idx.")

    return torch.where(force_start_mask, motion_start_idx[motion_ids], sampled_time_steps)


def motion_clip_timeout_mask(motion_finished: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
    """Return clip-boundary timeouts that do not overlap true terminations."""

    if motion_finished.shape != terminated.shape:
        raise ValueError(
            f"motion_finished and terminated must have equal shapes, got {motion_finished.shape} and "
            f"{terminated.shape}."
        )
    if motion_finished.dtype != torch.bool or terminated.dtype != torch.bool:
        raise TypeError("motion_finished and terminated must use torch.bool values.")
    return motion_finished & ~terminated


def adaptive_failure_mask(
    terminated: torch.Tensor,
    extra_failure_masks: Sequence[torch.Tensor] = (),
) -> torch.Tensor:
    """Combine physical terminations with explicitly configured adaptive failures."""

    if terminated.dtype != torch.bool:
        raise TypeError("terminated must use torch.bool values.")
    combined = terminated.clone()
    for failure_mask in extra_failure_masks:
        if failure_mask.shape != terminated.shape:
            raise ValueError(
                f"Every adaptive failure mask must have shape {terminated.shape}, got {failure_mask.shape}."
            )
        if failure_mask.dtype != torch.bool:
            raise TypeError("Every adaptive failure mask must use torch.bool values.")
        combined |= failure_mask
    return combined


class MultiMotionAdaptiveSampler:
    """Uniformly select a clip, then adaptively select a failure-prone phase within that clip."""

    def __init__(
        self,
        motion_start_idx: torch.Tensor,
        motion_end_idx: torch.Tensor,
        env_fps: float,
        device: str,
        adaptive_kernel_size: int = 1,
        adaptive_lambda: float = 0.8,
        adaptive_uniform_ratio: float = 0.1,
        adaptive_alpha: float = 0.001,
        motion_signatures: Sequence[str] | None = None,
    ):
        if motion_start_idx.ndim != 1 or motion_end_idx.shape != motion_start_idx.shape:
            raise ValueError("motion_start_idx and motion_end_idx must be one-dimensional tensors of equal shape.")
        if motion_start_idx.numel() == 0:
            raise ValueError("At least one motion is required.")
        if env_fps <= 0.0:
            raise ValueError(f"env_fps must be positive, got {env_fps}.")
        if adaptive_kernel_size < 1:
            raise ValueError(f"adaptive_kernel_size must be at least one, got {adaptive_kernel_size}.")
        if not 0.0 < adaptive_lambda <= 1.0:
            raise ValueError(f"adaptive_lambda must be in (0, 1], got {adaptive_lambda}.")
        if adaptive_uniform_ratio <= 0.0:
            raise ValueError(f"adaptive_uniform_ratio must be positive, got {adaptive_uniform_ratio}.")
        if not 0.0 < adaptive_alpha <= 1.0:
            raise ValueError(f"adaptive_alpha must be in (0, 1], got {adaptive_alpha}.")

        self.device = device
        self.motion_start_idx = motion_start_idx.to(device=device, dtype=torch.long)
        self.motion_end_idx = motion_end_idx.to(device=device, dtype=torch.long)
        self.motion_lengths = self.motion_end_idx - self.motion_start_idx
        if torch.any(self.motion_lengths < 2):
            raise ValueError("Every motion must contain at least two frames.")

        self.num_motions = self.motion_lengths.numel()
        if motion_signatures is not None and len(motion_signatures) != self.num_motions:
            raise ValueError(
                f"motion_signatures must contain one entry per motion, expected {self.num_motions}, "
                f"got {len(motion_signatures)}."
            )
        self.motion_signatures = None if motion_signatures is None else tuple(str(value) for value in motion_signatures)
        self.bin_counts = torch.floor(self.motion_lengths.float() / env_fps).long() + 1
        self._bin_counts_list = self.bin_counts.cpu().tolist()
        self.max_bin_count = int(self.bin_counts.max().item())
        self.adaptive_uniform_ratio = adaptive_uniform_ratio
        self.adaptive_alpha = adaptive_alpha

        kernel = torch.tensor(
            [adaptive_lambda**index for index in range(adaptive_kernel_size)],
            dtype=torch.float32,
            device=device,
        )
        self.kernel = kernel / kernel.sum()

        bin_ids = torch.arange(self.max_bin_count, device=device).unsqueeze(0)
        self.valid_bin_mask = bin_ids < self.bin_counts.unsqueeze(1)
        self.bin_failed_count = torch.zeros((self.num_motions, self.max_bin_count), dtype=torch.float32, device=device)
        self._current_bin_failed = torch.zeros_like(self.bin_failed_count)

    def state_dict(self) -> dict[str, torch.Tensor | tuple[str, ...] | int | None]:
        """Return a portable checkpoint of the learned phase distribution.

        Motion boundaries are included as an identity contract. This prevents
        silently applying failure statistics to a different set or ordering of
        NPZ clips when training is resumed.
        """

        return {
            "version": 1,
            "motion_start_idx": self.motion_start_idx.detach().cpu().clone(),
            "motion_end_idx": self.motion_end_idx.detach().cpu().clone(),
            "bin_counts": self.bin_counts.detach().cpu().clone(),
            "motion_signatures": self.motion_signatures,
            "bin_failed_count": self.bin_failed_count.detach().cpu().clone(),
            "current_bin_failed": self._current_bin_failed.detach().cpu().clone(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore adaptive statistics after validating the motion topology."""

        required_keys = {
            "version",
            "motion_start_idx",
            "motion_end_idx",
            "bin_counts",
            "motion_signatures",
            "bin_failed_count",
            "current_bin_failed",
        }
        missing_keys = sorted(required_keys - set(state_dict))
        if missing_keys:
            raise KeyError(f"Motion sampler checkpoint is missing keys: {missing_keys}.")
        if state_dict["version"] != 1:
            raise ValueError(f"Unsupported motion sampler checkpoint version: {state_dict['version']!r}.")

        topology = {
            "motion_start_idx": self.motion_start_idx,
            "motion_end_idx": self.motion_end_idx,
            "bin_counts": self.bin_counts,
        }
        for name, expected in topology.items():
            restored = torch.as_tensor(state_dict[name], dtype=torch.long, device=self.device)
            if restored.shape != expected.shape or not torch.equal(restored, expected):
                raise ValueError(
                    f"Motion sampler checkpoint {name} does not match the configured NPZ topology: "
                    f"expected {expected.detach().cpu().tolist()}, got {restored.detach().cpu().tolist()}."
                )

        restored_signatures = state_dict["motion_signatures"]
        if restored_signatures is not None:
            if not isinstance(restored_signatures, (list, tuple)):
                raise TypeError("Motion sampler checkpoint motion_signatures must be a sequence of strings or None.")
            restored_signatures = tuple(str(value) for value in restored_signatures)
        if restored_signatures != self.motion_signatures:
            raise ValueError(
                "Motion sampler checkpoint NPZ signatures do not match the configured motion files: "
                f"expected {self.motion_signatures}, got {restored_signatures}."
            )

        restored_statistics = {}
        for name in ("bin_failed_count", "current_bin_failed"):
            restored = torch.as_tensor(state_dict[name], dtype=torch.float32, device=self.device)
            if restored.shape != self.bin_failed_count.shape:
                raise ValueError(
                    f"Motion sampler checkpoint {name} must have shape {tuple(self.bin_failed_count.shape)}, "
                    f"got {tuple(restored.shape)}."
                )
            if not torch.isfinite(restored).all() or torch.any(restored < 0.0):
                raise ValueError(f"Motion sampler checkpoint {name} must contain finite non-negative values.")
            if torch.any(restored[~self.valid_bin_mask] != 0.0):
                raise ValueError(f"Motion sampler checkpoint {name} contains data in an invalid phase bin.")
            restored_statistics[name] = restored

        self.bin_failed_count.copy_(restored_statistics["bin_failed_count"])
        self._current_bin_failed.copy_(restored_statistics["current_bin_failed"])

    def record_failures(self, motion_ids: torch.Tensor, global_time_steps: torch.Tensor):
        """Accumulate failures at their original clip-local phase."""

        if motion_ids.numel() == 0:
            return
        motion_ids = motion_ids.long()
        local_time_steps = global_time_steps.long() - self.motion_start_idx[motion_ids]
        local_time_steps = torch.minimum(torch.clamp(local_time_steps, min=0), self.motion_lengths[motion_ids] - 1)
        failed_bins = torch.minimum(
            (local_time_steps * self.bin_counts[motion_ids]) // self.motion_lengths[motion_ids],
            self.bin_counts[motion_ids] - 1,
        )
        flat_indices = motion_ids * self.max_bin_count + failed_bins
        increments = torch.bincount(flat_indices, minlength=self.num_motions * self.max_bin_count)
        self._current_bin_failed += increments.view(self.num_motions, self.max_bin_count).float()

    @property
    def phase_sampling_probabilities(self) -> torch.Tensor:
        probabilities = torch.zeros_like(self.bin_failed_count)
        for motion_id, bin_count in enumerate(self._bin_counts_list):
            motion_probabilities = self.bin_failed_count[motion_id, :bin_count]
            motion_probabilities = motion_probabilities + self.adaptive_uniform_ratio / float(bin_count)
            motion_probabilities = F.pad(
                motion_probabilities.view(1, 1, -1),
                (0, self.kernel.numel() - 1),
                mode="replicate",
            )
            motion_probabilities = F.conv1d(motion_probabilities, self.kernel.view(1, 1, -1)).view(-1)
            probabilities[motion_id, :bin_count] = motion_probabilities / motion_probabilities.sum()
        return probabilities

    def sample(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return uniformly sampled motion IDs and adaptive global start-frame indices."""

        if num_samples < 0:
            raise ValueError(f"num_samples must be non-negative, got {num_samples}.")
        if num_samples == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty

        motion_ids = torch.randint(0, self.num_motions, (num_samples,), device=self.device)
        phase_probabilities = self.phase_sampling_probabilities[motion_ids]
        sampled_bins = torch.multinomial(phase_probabilities, 1, replacement=True).squeeze(1)

        phase = (sampled_bins.float() + torch.rand(num_samples, device=self.device)) / self.bin_counts[
            motion_ids
        ].float()
        local_time_steps = (phase * (self.motion_lengths[motion_ids] - 1).float()).long()
        local_time_steps = torch.minimum(local_time_steps, self.motion_lengths[motion_ids] - 2)
        global_time_steps = self.motion_start_idx[motion_ids] + local_time_steps
        return motion_ids, global_time_steps

    def sample_uniform(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniformly sample a clip and a valid non-final start frame within it."""

        if num_samples < 0:
            raise ValueError(f"num_samples must be non-negative, got {num_samples}.")
        if num_samples == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty

        motion_ids = torch.randint(0, self.num_motions, (num_samples,), device=self.device)
        local_time_steps = (
            torch.rand(num_samples, device=self.device) * (self.motion_lengths[motion_ids] - 1).float()
        ).long()
        return motion_ids, self.motion_start_idx[motion_ids] + local_time_steps

    def update(self):
        self.bin_failed_count = (
            self.adaptive_alpha * self._current_bin_failed + (1.0 - self.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def get_metrics(self) -> dict[str, torch.Tensor]:
        probabilities = self.phase_sampling_probabilities
        entropies = []
        top_probabilities = []
        top_bins = []
        for motion_id, bin_count in enumerate(self._bin_counts_list):
            motion_probabilities = probabilities[motion_id, :bin_count]
            entropy = -(motion_probabilities * (motion_probabilities + 1.0e-12).log()).sum()
            if bin_count > 1:
                entropy = entropy / np.log(bin_count)
            else:
                entropy = torch.ones_like(entropy)
            top_probability, top_bin = motion_probabilities.max(dim=0)
            entropies.append(entropy)
            top_probabilities.append(top_probability)
            top_bins.append(top_bin.float() / float(bin_count))

        return {
            "sampling_entropy": torch.stack(entropies).mean(),
            "sampling_top1_prob": torch.stack(top_probabilities).mean(),
            "sampling_top1_bin": torch.stack(top_bins).mean(),
        }
