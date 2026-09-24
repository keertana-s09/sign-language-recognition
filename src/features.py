"""Feature variants built on top of the stored landmark vectors.

extract_landmarks.py saves each hand as 63 floats (one-hand ASL) or 127 floats
(two-hand ISL: [primary 63][secondary 63][present flag]). Translation and scale
are already handled by normalize/normalize_two - but *rotation* and *anatomy*
are not, and those are exactly what differ between one person's hands and
another's.

Everything here is a pure function of those stored vectors, so feature ideas
can be tested against existing .npz files without re-running MediaPipe.

One-hand (63):
  raw       63   coordinates only
  rot       63   rotation-invariant canonical frame
  dist     210   all pairwise point distances
  rot+dist 273
  raw+dist 273

Two-hand (127):
  raw       127  coordinates + present flag
  dist      421  210 per hand + present flag
  raw+dist  547  127 + 420
  cross     862  210 per hand + 441 cross-hand + present flag
  raw+cross 988  127 + 420 + 441
"""

import numpy as np

N_LANDMARKS = 21
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17

MODES_1H = ("raw", "rot", "dist", "rot+dist", "raw+dist")
MODES_2H = ("raw", "dist", "raw+dist", "cross", "raw+cross")
MODES = tuple(sorted(set(MODES_1H + MODES_2H)))

_IU, _JU = np.triu_indices(N_LANDMARKS, k=1)


def _as_points(X):
    """(N, 63) -> (N, 21, 3)"""
    return np.asarray(X, dtype=np.float32).reshape(-1, N_LANDMARKS, 3)


def canonical_rotation(points):
    """Re-express each hand in a self-anchored frame. One-hand only."""
    e1 = points[:, MIDDLE_MCP] - points[:, WRIST]
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True) + 1e-8

    across = points[:, INDEX_MCP] - points[:, PINKY_MCP]
    e2 = across - (across * e1).sum(1, keepdims=True) * e1
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True) + 1e-8

    e3 = np.cross(e1, e2)

    basis = np.stack([e1, e2, e3], axis=1)
    return np.einsum("nij,nkj->nki", basis, points)


def pairwise_distances(points):
    """The 210 inter-point distances for one hand."""
    diff = points[:, _IU] - points[:, _JU]
    return np.linalg.norm(diff, axis=2)


def cross_hand_distances(primary, secondary):
    """All 21x21 = 441 distances between points on different hands.

    This is what separates two-handed letters: "right index touching left
    palm" is a specific pattern of small cross-hand distances. Within-hand
    distances can't see it.
    """
    diff = primary[:, :, None, :] - secondary[:, None, :, :]
    return np.linalg.norm(diff, axis=3).reshape(len(primary), -1)


def _split_two_hand(X):
    """(N, 127) -> primary (N,21,3), secondary (N,21,3), present (N,1)"""
    X = np.asarray(X, dtype=np.float32)
    primary = X[:, :63].reshape(-1, N_LANDMARKS, 3)
    secondary = X[:, 63:126].reshape(-1, N_LANDMARKS, 3)
    present = X[:, 126:127]
    return primary, secondary, present


def transform(X, mode):
    X = np.asarray(X, dtype=np.float32)
    width = X.shape[1]

    if width == N_LANDMARKS * 3:
        return _transform_1h(X, mode)
    if width == N_LANDMARKS * 3 * 2 + 1:
        return _transform_2h(X, mode)
    raise ValueError(f"expected 63 (one-hand) or 127 (two-hand), got {width}")


def _transform_1h(X, mode):
    if mode not in MODES_1H:
        raise ValueError(f"mode {mode!r} not valid for one-hand data, "
                         f"expected one of {MODES_1H}")
    points = _as_points(X)
    if mode == "raw":
        return X
    if mode == "rot":
        return canonical_rotation(points).reshape(len(X), -1)
    if mode == "dist":
        return pairwise_distances(points)
    if mode == "rot+dist":
        return np.hstack([canonical_rotation(points).reshape(len(X), -1),
                          pairwise_distances(points)])
    if mode == "raw+dist":
        return np.hstack([X, pairwise_distances(points)])
    raise ValueError(f"unknown mode {mode!r}")


def _transform_2h(X, mode):
    if mode not in MODES_2H:
        raise ValueError(f"mode {mode!r} not valid for two-hand data, "
                         f"expected one of {MODES_2H}")

    primary, secondary, present = _split_two_hand(X)

    if mode == "raw":
        return X

    d_primary = pairwise_distances(primary)
    d_secondary = pairwise_distances(secondary)

    if mode == "dist":
        return np.hstack([d_primary, d_secondary, present])

    if mode == "raw+dist":
        return np.hstack([X, d_primary, d_secondary])

    d_cross = cross_hand_distances(primary, secondary)
    # Zero out cross-hand distances when only one hand is present, so the
    # model doesn't see phantom relationships with the zero-filled secondary.
    d_cross = d_cross * present

    if mode == "cross":
        return np.hstack([d_primary, d_secondary, d_cross, present])

    if mode == "raw+cross":
        return np.hstack([X, d_primary, d_secondary, d_cross])

    raise ValueError(f"unknown mode {mode!r}")


def augment(X, y, groups, factor, seed=0):
    """Synthesize extra training hands by rotating and jittering real ones."""
    if factor <= 1:
        return X, y, groups

    width = X.shape[1]
    rng = np.random.RandomState(seed)

    if width == N_LANDMARKS * 3:
        return _augment_1h(X, y, groups, factor, rng)
    if width == N_LANDMARKS * 3 * 2 + 1:
        return _augment_2h(X, y, groups, factor, rng)
    raise ValueError(f"expected 63 or 127, got {width}")


def _random_rotation(rng, n):
    angles = rng.normal(0, 0.18, size=(n, 3))
    cos, sin = np.cos(angles), np.sin(angles)
    rot = np.zeros((n, 3, 3), dtype=np.float32)
    rot[:, 0, 0] = cos[:, 1] * cos[:, 2]
    rot[:, 0, 1] = -cos[:, 1] * sin[:, 2]
    rot[:, 0, 2] = sin[:, 1]
    rot[:, 1, 0] = cos[:, 0] * sin[:, 2] + sin[:, 0] * sin[:, 1] * cos[:, 2]
    rot[:, 1, 1] = cos[:, 0] * cos[:, 2] - sin[:, 0] * sin[:, 1] * sin[:, 2]
    rot[:, 1, 2] = -sin[:, 0] * cos[:, 1]
    rot[:, 2, 0] = sin[:, 0] * sin[:, 2] - cos[:, 0] * sin[:, 1] * cos[:, 2]
    rot[:, 2, 1] = sin[:, 0] * cos[:, 2] + cos[:, 0] * sin[:, 1] * sin[:, 2]
    rot[:, 2, 2] = cos[:, 0] * cos[:, 1]
    return rot


def _jitter_points(rng, points):
    scale = rng.uniform(0.92, 1.08, size=(len(points), 1, 1))
    noise = rng.normal(0, 0.02, size=points.shape)
    return points * scale + noise


def _augment_1h(X, y, groups, factor, rng):
    points = _as_points(X)
    extra = []
    for _ in range(factor - 1):
        rot = _random_rotation(rng, len(points))
        turned = np.einsum("nij,nkj->nki", rot, points)
        turned = _jitter_points(rng, turned)
        extra.append(turned.reshape(len(X), -1).astype(np.float32))
    return np.concatenate([X] + extra), np.tile(y, factor), np.tile(groups, factor)


def _augment_2h(X, y, groups, factor, rng):
    """Augment two-hand ISL vectors.

    Both hands share one coordinate frame anchored on the primary hand, so
    rotating the primary hand must drag the secondary with it - otherwise the
    augmented data would have the hands in impossible relative positions.
    """
    primary, secondary, present = _split_two_hand(X)
    extra = []
    for _ in range(factor - 1):
        rot = _random_rotation(rng, len(primary))
        p_turned = _jitter_points(rng, np.einsum("nij,nkj->nki", rot, primary))
        s_turned = _jitter_points(rng, np.einsum("nij,nkj->nki", rot, secondary))
        # Zero-fill secondary when it wasn't present in the original.
        s_turned = s_turned * present[:, :, None]
        vec = np.zeros((len(X), 127), dtype=np.float32)
        vec[:, :63] = p_turned.reshape(-1, 63)
        vec[:, 63:126] = s_turned.reshape(-1, 63)
        vec[:, 126] = present[:, 0]
        extra.append(vec)
    return np.concatenate([X] + extra), np.tile(y, factor), np.tile(groups, factor)
