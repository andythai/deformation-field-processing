"""3D (SimplexConstraint3D) path of the windowed engine — phase 1 of the 3D port.

Every test here is additive: the 2D families are covered by the existing
windowed suites and by benchmarks/windowed_2d_identity.py (byte-identity).
"""

import numpy as np
import pytest

from dvfopt.constraints import SimplexConstraint3D
from dvfopt.core.primitives import isqp as isqp_mod
from dvfopt.core.windowed import (
    LOCALITY,
    build_subproblem,
    find_windows,
    min_field,
    windowed_correct,
)
from dvfopt.objectives import L2Objective, NoneObjective

needs_osqp = pytest.mark.skipif(not isqp_mod.HAS_OSQP, reason="osqp not installed")

THR = 0.01


def test_box_helpers_are_dimension_agnostic():
    from dvfopt.core.windowed._common import _box_size, _box_slices, _pad_box

    assert _box_slices((1, 4, 2, 6)) == (slice(1, 4), slice(2, 6))
    assert _box_slices((0, 2, 1, 4, 2, 6)) == (slice(0, 2), slice(1, 4), slice(2, 6))
    assert _box_size((1, 4, 2, 6)) == 12
    assert _box_size((0, 2, 1, 4, 2, 6)) == 24
    assert _pad_box((1, 4, 2, 6), (5, 8), 2) == (0, 5, 0, 8)
    assert _pad_box((2, 4, 3, 5, 4, 6), (10, 10, 10), 1) == (1, 5, 2, 6, 3, 7)


def test_find_windows_3d_boxes_and_border_rule():
    mask = np.zeros((10, 12, 14), bool)
    mask[4, 5, 6] = True  # interior fold
    mask[0, 1, 1] = True  # fold on the z=0 face (and near the y/x=0 faces)
    boxes = sorted(find_windows(mask, margin=2, ring=1))
    # dilation by margin+ring=3 is an L1 ball; the bbox is inset by ring except on a
    # side that reached the volume border, where the fold must stay free
    assert boxes == [(0, 3, 0, 4, 0, 4), (2, 7, 3, 8, 4, 9)]


def test_find_windows_2d_is_unchanged():
    mask = np.zeros((20, 20), bool)
    mask[10, 10] = True
    assert find_windows(mask, margin=3, ring=1) == [(7, 14, 7, 14)]
