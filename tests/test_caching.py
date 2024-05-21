# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the caching infrastructure.
."""

from __future__ import annotations

import itertools as it

import jax
import numpy as np
import pytest

import jace
from jace.jax import stages


@pytest.fixture(autouse=True)
def _clear_translation_cache():
    """Decorator that clears the translation cache.

    Ensures that a function finds an empty cache and clears up afterwards.

    Todo:
        Ask Enrique how I can make that fixture apply everywhere not just in the file but the whole test suite.
    """
    from jace.jax import translation_cache as tcache

    tcache.clear_translation_cache()
    yield
    tcache.clear_translation_cache()


def test_caching_same_sizes():
    """The behaviour of the cache if same sizes are used."""
    jax.config.update("jax_enable_x64", True)

    # Counter for how many time it was lowered.
    lowering_cnt = [0]

    # This is the pure Python function.
    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A * B

    # this is the wrapped function.
    @jace.jit
    def wrapped(A, B):
        lowering_cnt[0] += 1
        return testee(A, B)

    # First batch of arguments.
    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    # The second batch of argument, it is the same size (structurally) but different values.
    AA = A + 1.0362
    BB = B + 0.638956

    # Now let's lower it once directly and call it.
    lowered: stages.JaceLowered = wrapped.lower(A, B)
    compiled: stages.JaceCompiled = lowered.compile()
    assert lowering_cnt[0] == 1
    assert np.allclose(testee(A, B), compiled(A, B))

    # Now lets call the wrapped object directly, since we already did the lowering
    #  no longering (and compiling) is needed.
    assert np.allclose(testee(A, B), wrapped(A, B))
    assert lowering_cnt[0] == 1

    # Now lets call it with different objects, that have the same structure.
    #  Again no lowering should happen.
    assert np.allclose(testee(AA, BB), wrapped(AA, BB))
    assert wrapped.lower(AA, BB) is lowered
    assert wrapped.lower(A, B) is lowered
    assert lowering_cnt[0] == 1


def test_caching_different_sizes():
    """The behaviour of the cache if different sizes where used."""
    jax.config.update("jax_enable_x64", True)

    # Counter for how many time it was lowered.
    lowering_cnt = [0]

    # This is the wrapped function.
    @jace.jit
    def wrapped(A, B):
        lowering_cnt[0] += 1
        return A * B

    # First size of arguments
    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    # Second size of arguments
    C = np.arange(16, dtype=np.float64).reshape((4, 4))
    D = np.full((4, 4), 10, dtype=np.float64)

    # Now lower the function once for each.
    lowered1 = wrapped.lower(A, B)
    lowered2 = wrapped.lower(C, D)
    assert lowering_cnt[0] == 2
    assert lowered1 is not lowered2

    # Now also check if the compilation works as intended
    compiled1 = lowered1.compile()
    compiled2 = lowered2.compile()
    assert lowering_cnt[0] == 2
    assert compiled1 is not compiled2


@pytest.mark.skip(reason="Missing primitive translators")
def test_caching_different_structure():
    """Now tests if we can handle multiple arguments with different structures.

    Todo:
        - Extend with strides once they are part of the cache.
    """
    jax.config.update("jax_enable_x64", True)

    # This is the wrapped function.
    lowering_cnt = [0]

    @jace.jit
    def wrapped(A, B):
        lowering_cnt[0] += 1
        return A * 4.0, B + 2.0

    A = np.full((4, 30), 10, dtype=np.float64)
    B = np.full((4, 3), 10, dtype=np.float64)
    C = np.full((5, 3), 14, dtype=np.float64)
    D = np.full((6, 3), 14, dtype=np.int64)

    # These are the arrays.
    args: dict[int, np.ndarray] = {id(x): x for x in [A, B, C, D]}
    # These are the known lowerings.
    lowerings: dict[tuple[int, int], stages.JaceLowered] = {}
    lowering_ids: set[int] = set()

    # Generating the lowerings
    for arg1, arg2 in it.permutations([A, B, C, D], 2):
        lower = wrapped.lower(arg1, arg2)
        assert id(lower) not in lowering_ids
        lowerings[id(arg1), id(arg2)] = lower
        lowering_ids.add(id(lower))

    # Now check if they are still cached.
    for arg1, arg2 in it.permutations([A, B, C, D], 2):
        lower = wrapped.lower(arg1, arg2)
        clower = lowerings[id(arg1), id(arg2)]
        assert clower is lower


def test_caching_compilation():
    """Tests the compilation cache, this is just very simple, since it uses the same code paths as lowering."""
    jax.config.update("jax_enable_x64", True)

    @jace.jit
    def jaceWrapped(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        C = A * B
        D = C + A
        E = D + B  # Just enough state.
        return A + B + C + D + E

    # These are the argument
    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    # Now we lower it.
    jaceLowered = jaceWrapped.lower(A, B)

    # Now we compile it with enabled optimization.
    optiCompiled = jaceLowered.compile(stages.JaceLowered.DEF_COMPILER_OPTIONS)

    # Passing `None` also means 'default' which is a bit strange, but it is what Jax does.
    assert optiCompiled is jaceLowered.compile(None)

    # Now we compile it without any optimization.
    unoptiCompiled = jaceLowered.compile({})

    # Because of the way how things work the optimized must have more than the unoptimized.
    #  If there is sharing, then this would not be the case.
    assert optiCompiled._csdfg.sdfg.number_of_nodes() == 1
    assert optiCompiled._csdfg.sdfg.number_of_nodes() < unoptiCompiled._csdfg.sdfg.number_of_nodes()