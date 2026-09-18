"""Mass-preserving transformations for variable-resolution categorical beliefs."""

from __future__ import annotations

import numpy as np


def _normalized_square(values: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size != size**2:
        raise ValueError(f"posterior must contain exactly {size**2} values")
    array = array.reshape(size, size)
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError("posterior values must be finite and nonnegative")
    total = float(array.sum())
    if total <= 0:
        raise ValueError("posterior must have positive mass")
    return array / total


def canonicalize_posterior(
    posterior: np.ndarray,
    resolution: int,
    canonical_size: int = 20,
) -> np.ndarray:
    """Expand a native posterior to a canonical grid without inventing mass."""

    if canonical_size % resolution:
        raise ValueError("resolution must divide canonical_size")
    native = _normalized_square(posterior, resolution)
    scale = canonical_size // resolution
    canonical = np.repeat(np.repeat(native, scale, axis=0), scale, axis=1)
    canonical /= scale**2
    return canonical


def project_canonical(canonical: np.ndarray, target_resolution: int) -> np.ndarray:
    """Sum canonical probability mass into a lower-resolution posterior."""

    values = np.asarray(canonical, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("canonical posterior must be a square matrix")
    canonical_size = values.shape[0]
    if canonical_size % target_resolution:
        raise ValueError("target_resolution must divide the canonical size")
    canonical_values = _normalized_square(values, canonical_size)
    scale = canonical_size // target_resolution
    return canonical_values.reshape(target_resolution, scale, target_resolution, scale).sum(
        axis=(1, 3)
    )


def remap_posterior(
    posterior: np.ndarray,
    source_resolution: int,
    target_resolution: int,
    canonical_size: int = 20,
) -> np.ndarray:
    """Transfer a belief between supported grids through a canonical measure."""

    canonical = canonicalize_posterior(posterior, source_resolution, canonical_size)
    return project_canonical(canonical, target_resolution)


def normalized_entropy(posterior: np.ndarray, resolution: int) -> float:
    """Return entropy as a fraction of the native representation's maximum."""

    values = _normalized_square(posterior, resolution).ravel()
    positive = values[values > 0]
    entropy = float(-positive.dot(np.log(positive)))
    maximum = float(np.log(resolution**2))
    return 0.0 if maximum == 0 else entropy / maximum


def information_loss(
    posterior: np.ndarray,
    source_resolution: int,
    target_resolution: int,
    canonical_size: int = 20,
) -> float:
    """Measure detail destroyed by a resolution switch using normalized JS divergence."""

    original = canonicalize_posterior(posterior, source_resolution, canonical_size).ravel()
    compressed = project_canonical(
        original.reshape(canonical_size, canonical_size), target_resolution
    )
    reconstructed = canonicalize_posterior(
        compressed,
        target_resolution,
        canonical_size,
    ).ravel()
    midpoint = 0.5 * (original + reconstructed)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        positive = left > 0
        return float(np.sum(left[positive] * np.log(left[positive] / right[positive])))

    js = 0.5 * kl(original, midpoint) + 0.5 * kl(reconstructed, midpoint)
    return js / np.log(2.0)
