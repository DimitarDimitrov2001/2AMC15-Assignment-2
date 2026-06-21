from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class StateNormalizer:
    """Normalize ContinuousEnvironment observations without adding shortcut features."""

    grid_shape: tuple[int, int]
    max_sensor_range: float

    def __post_init__(self) -> None:
        if len(self.grid_shape) != 2:
            raise ValueError("grid_shape must be a pair of integers")
        if self.grid_shape[0] <= 0 or self.grid_shape[1] <= 0:
            raise ValueError("grid dimensions must be positive")
        if self.max_sensor_range <= 0:
            raise ValueError("max_sensor_range must be positive")

    def normalize(self, state: np.ndarray) -> np.ndarray:
        state_arr = np.asarray(state, dtype=np.float32).copy()
        if state_arr.ndim != 1 or state_arr.shape[0] < 3:
            raise ValueError("state must be a 1D vector with at least x, y, theta")

        rows, cols = self.grid_shape
        state_arr[0] = state_arr[0] / max(1.0, float(rows - 1))
        state_arr[1] = state_arr[1] / max(1.0, float(cols - 1))
        state_arr[2] = (state_arr[2] % 360.0) / 360.0
        if state_arr.shape[0] > 3:
            state_arr[3:] = state_arr[3:] / float(self.max_sensor_range)
        return np.clip(state_arr, 0.0, 1.0).astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["grid_shape"] = list(self.grid_shape)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "StateNormalizer":
        return cls(
            grid_shape=tuple(payload["grid_shape"]),  # type: ignore[arg-type]
            max_sensor_range=float(payload["max_sensor_range"]),
        )
