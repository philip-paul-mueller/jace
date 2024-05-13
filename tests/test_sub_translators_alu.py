# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests for the ALU translator."""

from __future__ import annotations

import jax
import numpy as np

from jace import util as jutil


def test_add():
    """Simple add function."""
    jax.config.update("jax_enable_x64", True)

    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    ref = testee(A, B)
    res = jutil._jace_run(testee, A, B)

    assert np.allclose(ref, res), f"Expected '{ref}' got '{res}'."


def test_add2():
    """Simple add function, with literal."""
    jax.config.update("jax_enable_x64", True)

    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        c = A + 0.01
        d = B * 0.6
        e = c / 1.0
        f = d - 0.1
        return e + f * d

    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    ref = testee(A, B)
    res = jutil._jace_run(testee, A, B)

    assert np.allclose(ref, res), f"Expected '{ref}' got '{res}'."


def test_add3():
    """Simple add function, with constant."""
    jax.config.update("jax_enable_x64", True)

    def testee(A: np.ndarray) -> np.ndarray:
        return A + jax.numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    A = np.ones((3, 3), dtype=np.float64)

    ref = testee(A)
    res = jutil._jace_run(testee, A)

    assert np.allclose(ref, res), f"Expected '{ref}' got '{res}'."