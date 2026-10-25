import numpy as np


def scan_to_observation(
    ranges: list[float] | tuple[float, ...] | np.ndarray,
    range_min: float,
    range_max: float,
    bins: int,
) -> np.ndarray:
    """Pool a laser scan into a fixed-size, normalized observation."""
    scan = np.asarray(ranges, dtype=np.float32)
    if scan.ndim != 1 or scan.size == 0:
        raise ValueError("Laser scan must contain at least one range value.")
    if not np.isfinite(range_min) or not np.isfinite(range_max) or range_max <= 0:
        raise ValueError("Laser scan range limits must be finite and range_max positive.")
    if bins <= 0:
        raise ValueError("bins must be a positive integer.")

    scan = np.nan_to_num(scan, nan=range_min, posinf=range_max, neginf=range_min)
    scan = np.clip(scan, max(0.0, range_min), range_max)

    if scan.size >= bins:
        pooled = np.asarray([chunk.min() for chunk in np.array_split(scan, bins)])
    else:
        pooled = np.interp(
            np.linspace(0, scan.size - 1, bins),
            np.arange(scan.size),
            scan,
        )

    return np.asarray(pooled / range_max, dtype=np.float32)
