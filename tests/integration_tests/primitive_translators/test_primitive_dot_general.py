# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

import jace

from tests import util as testutil


@pytest.fixture(
    params=[
        ((20, 20), (20, 20)),
        ((20, 15), (15, 20)),
        ((20), (20)),
        ((20, 10), (10)),
    ]
)
def simple_shapes(request) -> tuple[Sequence[int], Sequence[int]]:
    """
    Shapes for the `test_dot_general()` test, first the lhs, followed by the rhs shape.
    """
    return request.param


@pytest.fixture(params=[("C", "C"), ("C", "F"), ("F", "C"), ("F", "F")])
def mem_orders(request):
    """Memory order for the lhs and the rhs."""
    return request.param


def test_dot_general(
    simple_shapes: tuple[Sequence[int], Sequence[int]],
    mem_orders: tuple[str, str],
):
    @jace.jit
    def testee(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return lhs @ rhs

    shape_lhs, shape_rhs = simple_shapes
    lhs = testutil.make_array(shape=shape_lhs, order=mem_orders[0])
    rhs = testutil.make_array(shape=shape_rhs, order=mem_orders[1])
    ref = lhs @ rhs
    res = testee(lhs, rhs)

    assert np.allclose(ref, res), f"Expected '{ref}' but got '{res}'."
