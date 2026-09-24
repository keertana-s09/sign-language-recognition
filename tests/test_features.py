"""Invariance tests for the feature pipeline.

These assert the *claims* the project makes about its features, not merely that
the functions run. If normalization silently stops being scale-invariant, live
accuracy degrades in a way that no accuracy metric would attribute to the real
cause - so the properties are worth pinning down directly.
"""

import numpy as np
import pytest

import features as F
from extract_landmarks import normalize, normalize_two

rng = np.random.RandomState(0)


def random_hand():
    """A plausible hand: wrist at origin, middle knuckle roughly one unit away."""
    hand = rng.normal(0, 0.4, size=(F.N_LANDMARKS, 3)).astype(np.float32)
    hand[F.WRIST] = 0.0
    hand[F.MIDDLE_MCP] = [0.0, 1.0, 0.0]
    return hand


def rotation_matrix(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    return (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])).astype(np.float32)


class TestNormalize:
    def test_translation_invariant(self):
        hand = random_hand()
        moved = hand + np.array([3.0, -2.0, 0.5], dtype=np.float32)
        assert np.allclose(normalize(hand), normalize(moved), atol=1e-5)

    def test_scale_invariant(self):
        """Moving closer to the camera must not change the features."""
        hand = random_hand()
        assert np.allclose(normalize(hand), normalize(hand * 2.5), atol=1e-5)

    def test_wrist_at_origin(self):
        out = normalize(random_hand()).reshape(F.N_LANDMARKS, 3)
        assert np.allclose(out[F.WRIST], 0.0, atol=1e-6)

    def test_unit_reference_length(self):
        out = normalize(random_hand()).reshape(F.N_LANDMARKS, 3)
        assert np.isclose(np.linalg.norm(out[F.MIDDLE_MCP]), 1.0, atol=1e-5)

    def test_degenerate_hand_rejected(self):
        """All points collapsed: scale is zero and there is nothing to divide by."""
        assert normalize(np.zeros((F.N_LANDMARKS, 3), dtype=np.float32)) is None

    def test_output_width(self):
        assert normalize(random_hand()).shape == (63,)


class TestCanonicalRotation:
    def test_rotation_invariant(self):
        """The central claim of the `rot` feature set.

        Turning the hand in 3D must produce identical canonical coordinates.
        """
        hand = random_hand()
        turned = hand @ rotation_matrix(0.7, -1.1, 0.3).T
        a = F.canonical_rotation(hand[None])
        b = F.canonical_rotation(turned[None].astype(np.float32))
        assert np.allclose(a, b, atol=1e-4)

    def test_reference_axis_is_canonical(self):
        """Middle knuckle lands on +x by construction."""
        out = F.canonical_rotation(random_hand()[None])
        assert np.allclose(out[0, F.MIDDLE_MCP], [1.0, 0.0, 0.0], atol=1e-4)

    def test_basis_is_orthonormal(self):
        """A non-orthonormal basis would distort distances, not just rotate."""
        hand = random_hand()
        out = F.canonical_rotation(hand[None])[0]
        for i in range(F.N_LANDMARKS):
            assert np.isclose(np.linalg.norm(out[i]), np.linalg.norm(hand[i]), atol=1e-4)


class TestPairwiseDistances:
    def test_count(self):
        assert F.pairwise_distances(random_hand()[None]).shape == (1, 210)

    def test_rotation_invariant(self):
        hand = random_hand()
        turned = (hand @ rotation_matrix(0.4, 2.0, -0.9).T).astype(np.float32)
        assert np.allclose(F.pairwise_distances(hand[None]),
                           F.pairwise_distances(turned[None]), atol=1e-4)

    def test_non_negative(self):
        assert (F.pairwise_distances(random_hand()[None]) >= 0).all()


class TestTransform:
    @pytest.mark.parametrize("mode,width", [
        ("raw", 63), ("rot", 63), ("dist", 210),
        ("rot+dist", 273), ("raw+dist", 273),
    ])
    def test_widths(self, mode, width):
        X = np.stack([normalize(random_hand()) for _ in range(4)])
        assert F.transform(X, mode).shape == (4, width)

    def test_raw_is_passthrough(self):
        X = np.stack([normalize(random_hand()) for _ in range(3)])
        assert np.array_equal(F.transform(X, "raw"), X)

    def test_rejects_bad_width(self):
        with pytest.raises(ValueError, match="expected 63.*or 127"):
            F.transform(np.zeros((2, 50), dtype=np.float32), "dist")

    def test_rejects_invalid_1h_mode(self):
        with pytest.raises(ValueError, match="not valid for one-hand"):
            F.transform(np.zeros((2, 63), dtype=np.float32), "cross")

    def test_rejects_invalid_2h_mode(self):
        with pytest.raises(ValueError, match="not valid for two-hand"):
            F.transform(np.zeros((2, 127), dtype=np.float32), "rot")


class TestAugment:
    def test_originals_preserved_first(self):
        """Real samples must survive untouched, or the training set drifts."""
        X = np.stack([normalize(random_hand()) for _ in range(5)])
        y = np.arange(5)
        out_X, out_y, _ = F.augment(X, y, y, 3)
        assert np.array_equal(out_X[:5], X)
        assert np.array_equal(out_y[:5], y)

    def test_multiplies_count(self):
        X = np.stack([normalize(random_hand()) for _ in range(5)])
        y = np.arange(5)
        out_X, out_y, out_g = F.augment(X, y, y, 4)
        assert len(out_X) == len(out_y) == len(out_g) == 20

    def test_labels_track_samples(self):
        X = np.stack([normalize(random_hand()) for _ in range(5)])
        y = np.arange(5)
        _, out_y, _ = F.augment(X, y, y, 3)
        assert np.array_equal(out_y, np.tile(y, 3))

    def test_factor_one_is_noop(self):
        X = np.stack([normalize(random_hand()) for _ in range(3)])
        y = np.arange(3)
        out_X, out_y, _ = F.augment(X, y, y, 1)
        assert np.array_equal(out_X, X) and np.array_equal(out_y, y)

    def test_copies_differ_from_originals(self):
        """If augmentation changed nothing it would just be duplication, which
        this project already measured to be harmful."""
        X = np.stack([normalize(random_hand()) for _ in range(5)])
        y = np.arange(5)
        out_X, _, _ = F.augment(X, y, y, 2)
        assert not np.allclose(out_X[:5], out_X[5:], atol=1e-3)


class TestTwoHandEncoding:
    def test_single_hand_sets_absent_flag(self):
        out = normalize_two([(random_hand(), "Right")])
        assert out.shape == (127,)
        assert out[126] == 0.0
        assert np.allclose(out[63:126], 0.0)

    def test_two_hands_set_present_flag(self):
        out = normalize_two([(random_hand(), "Right"), (random_hand(), "Left")])
        assert out[126] == 1.0

    def test_right_hand_becomes_primary(self):
        """Primary must be chosen by handedness, never detection order - the
        roles differ in two-handed signs, so swapping them scrambles classes."""
        right, left = random_hand(), random_hand()
        a = normalize_two([(right, "Right"), (left, "Left")])
        b = normalize_two([(left, "Left"), (right, "Right")])
        assert np.allclose(a, b, atol=1e-5)

    def test_relative_position_preserved(self):
        """Both hands share one frame, so moving the second hand must change the
        encoding. Normalizing each hand separately would not."""
        primary = random_hand()
        near = normalize_two([(primary, "Right"), (random_hand(), "Left")])
        far = normalize_two([(primary, "Right"),
                             (random_hand() + np.float32([5, 0, 0]), "Left")])
        assert not np.allclose(near[63:126], far[63:126], atol=1e-3)

    def test_empty_input_rejected(self):
        assert normalize_two([]) is None


def _make_two_hand_vectors(n):
    """Build n random 127-float ISL-style vectors."""
    vecs = []
    for _ in range(n):
        primary = random_hand()
        secondary = random_hand()
        v = normalize_two([(primary, "Right"), (secondary, "Left")])
        vecs.append(v)
    return np.stack(vecs)


def _make_one_hand_vectors(n):
    """Build n random 127-float vectors with only primary hand."""
    vecs = []
    for _ in range(n):
        v = normalize_two([(random_hand(), "Right")])
        vecs.append(v)
    return np.stack(vecs)


class TestTwoHandTransform:
    @pytest.mark.parametrize("mode,width", [
        ("raw", 127), ("dist", 421), ("raw+dist", 547),
        ("cross", 862), ("raw+cross", 988),
    ])
    def test_widths(self, mode, width):
        X = _make_two_hand_vectors(4)
        assert F.transform(X, mode).shape == (4, width)

    def test_raw_is_passthrough(self):
        X = _make_two_hand_vectors(3)
        assert np.array_equal(F.transform(X, "raw"), X)

    def test_dist_symmetric_for_identical_hands(self):
        """When both hands are the same, their within-hand distances should match."""
        hand = random_hand()
        v = normalize_two([(hand, "Right"), (hand, "Left")])
        out = F.transform(v[None], "dist")
        d_primary = out[0, :210]
        d_secondary = out[0, 210:420]
        assert np.allclose(d_primary, d_secondary, atol=1e-4)

    def test_cross_distances_zero_when_one_hand(self):
        """Cross-hand distances must be zeroed when only primary is present,
        or the model sees phantom relationships with the zero-filled secondary."""
        X = _make_one_hand_vectors(3)
        out = F.transform(X, "cross")
        # cross distances are after 210 + 210, before the present flag
        cross_block = out[:, 420:861]
        assert np.allclose(cross_block, 0.0, atol=1e-6)

    def test_cross_distances_nonzero_when_two_hands(self):
        X = _make_two_hand_vectors(3)
        out = F.transform(X, "cross")
        cross_block = out[:, 420:861]
        assert not np.allclose(cross_block, 0.0, atol=1e-3)


class TestTwoHandAugment:
    def test_originals_preserved(self):
        X = _make_two_hand_vectors(5)
        y = np.arange(5)
        out_X, out_y, _ = F.augment(X, y, y, 3)
        assert np.array_equal(out_X[:5], X)

    def test_multiplies_count(self):
        X = _make_two_hand_vectors(5)
        y = np.arange(5)
        out_X, _, _ = F.augment(X, y, y, 3)
        assert len(out_X) == 15

    def test_present_flag_preserved(self):
        """Augmented one-hand vectors must keep the absent flag at zero."""
        X = _make_one_hand_vectors(5)
        y = np.arange(5)
        out_X, _, _ = F.augment(X, y, y, 3)
        assert (out_X[:, 126] == 0.0).all()

    def test_two_hand_present_flag_preserved(self):
        X = _make_two_hand_vectors(5)
        y = np.arange(5)
        out_X, _, _ = F.augment(X, y, y, 3)
        assert (out_X[:, 126] == 1.0).all()

    def test_copies_differ(self):
        X = _make_two_hand_vectors(5)
        y = np.arange(5)
        out_X, _, _ = F.augment(X, y, y, 2)
        assert not np.allclose(out_X[:5], out_X[5:], atol=1e-3)
