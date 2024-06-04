# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests the compatibility of the JaCe api to Jax."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jace
from jace import translator, util
from jace.translator import pre_post_translation as ptrans

from tests import util as testutil


def test_jit():
    """Simple add function."""

    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    A = testutil.mkarray((4, 3))
    B = testutil.mkarray((4, 3))

    jax_testee = jax.jit(testee)
    jace_testee = jace.jit(testee)

    assert util.is_jaxified(jax_testee)
    assert not util.is_jaxified(jace_testee)
    assert not util.is_jaceified(jax_testee)
    assert util.is_jaceified(jace_testee)

    ref = jax_testee(A, B)
    res = jace_testee(A, B)

    assert np.allclose(ref, res), f"Expected '{ref}' got '{res}'."


def test_composition_itself():
    """Tests if JaCe is composable with itself."""

    # Pure Python functions
    def f_ref(x):
        return jnp.sin(x)

    def df_ref(x):
        return jnp.cos(x)

    def ddf_ref(x):
        return -jnp.sin(x)

    # Annotated functions.

    @jace.jit
    def f(x):
        return f_ref(x)

    @jace.jit
    def df(x):
        return jace.grad(f)(x)

    @jace.jit
    @jace.grad
    def ddf(x):
        return df(x)

    assert all(util.is_jaceified(x) for x in [f, df, ddf])

    x = 1.0
    for fun, fref in zip([f, df, ddf], [f_ref, df_ref, ddf_ref]):
        ref = fref(x)
        res = fun(x)
        assert np.allclose(ref, res), f"f: Expected '{ref}', got '{res}'."


@pytest.mark.skip(reason="Nested Jaxpr are not handled.")
def test_composition_with_jax():
    """Tests if JaCe can interact with Jax and vice versa."""

    def base_fun(A, B, C):
        return A + B * jnp.sin(C) - A * B

    @jace.jit
    def jace_fun(A, B, C):
        return jax.jit(base_fun)(A, B, C)

    def jax_fun(A, B, C):
        return jace.jit(base_fun)(A, B, C)

    A, B, C = (testutil.mkarray((10, 3, 50)) for _ in range(3))

    assert np.allclose(jace_fun(A, B, C), jax_fun(A, B, C))


@pytest.mark.skip(reason="Nested Jaxpr are not handled.")
def test_composition_with_jax_2():
    """Second test if JaCe can interact with Jax and vice versa."""

    @jax.jit
    def f1_jax(A, B):
        return A + B

    assert util.is_jaxified(f1_jax)

    @jace.jit
    def f2_jace(A, B, C):
        return f1_jax(A, B) - C

    assert util.is_jaceified(f2_jace)

    @jax.jit
    def f3_jax(A, B, C, D):
        return f2_jace(A, B, C) * D

    assert util.is_jaxified(f3_jax)

    @jace.jit
    def f3_jace(A, B, C, D):
        return f3_jax(A, B, C, D)

    assert util.is_jaceified(f3_jace)

    A, B, C, D = (testutil.mkarray((10, 3, 50)) for _ in range(4))

    ref = ((A + B) - C) * D

    res_jax = f3_jax(A, B, C, D)
    res_jace = f3_jace(A, B, C, D)

    assert np.allclose(ref, res_jax), "Jax failed."
    assert np.allclose(ref, res_jace), "JaCe Failed."


def test_grad_annotation_direct():
    """Test if `jace.grad` works directly."""

    def f(x):
        return jnp.sin(jnp.exp(jnp.cos(x**2)))

    @jax.grad
    def jax_ddf(x):
        return jax.grad(f)(x)

    @jax.jit
    def jace_ddf(x):
        return jace.grad(jace.grad(f))(x)

    # These are the random numbers where we test
    Xs = (testutil.mkarray(10) - 0.5) * 10

    for i in range(Xs.shape[0]):
        x = Xs[i]
        res = jace_ddf(x)
        ref = jax_ddf(x)
        assert np.allclose(res, ref)


def test_grad_control_flow():
    """Tests if `grad` and controlflow works.

    This requirement is mentioned in `https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#python-control-flow-autodiff`.
    """

    @jace.grad
    def df(x):
        if x < 3:
            return 3.0 * x**2
        return -4 * x

    x1 = 2.0
    df_x1 = 6 * x1
    x2 = 4.0
    df_x2 = -4.0

    res_1 = df(x1)
    res_2 = df(x2)

    assert df(x1) == df_x1, f"Failed lower branch, expected '{df_x1}', got '{res_1}'."
    assert df(x2) == df_x2, f"Failed upper branch, expected '{df_x2}', got '{res_2}'."


def test_disabled_x64():
    """Tests the behaviour of the tool chain if x64 support is disabled.

    Notes:
        Once the x64 issue is resolved make this test a bit more useful.
    """
    from jax.experimental import disable_x64

    def testee(A: np.ndarray, B: np.float64) -> np.ndarray:
        return A + B

    A = testutil.mkarray((4, 3))
    B = np.float64(10.0)

    # Run them with disabled x64 support
    #  This is basically a reimplementation of the `JaCeWrapped.lower()` function.
    #  but we have to do it this way to disable the x64 mode in translation.
    with disable_x64():
        jaxpr = jax.make_jaxpr(testee)(A, B)

    builder = translator.JaxprTranslationBuilder(
        primitive_translators=translator.get_registered_primitive_translators(),
    )
    trans_ctx: translator.TranslationContext = builder.translate_jaxpr(jaxpr)

    tsdfg: translator.TranslatedJaxprSDFG = ptrans.postprocess_jaxpr_sdfg(
        trans_ctx=trans_ctx,
        fun=testee,
        call_args=(A, B),  # Already linearised, since we only accept positional args.
        intree=None,  # Not yet implemented.
    )

    # Because x64 is disabled Jax traces the input as float32, even if we have passed
    #  float64 as input! Calling the resulting SDFG with the arguments we used for lowering
    #  will result in an error, because of the situation, `sizeof(float32) < sizeof(float64)`,
    #  no out of bound error would result, but the values are garbage.
    assert tsdfg.sdfg.arrays[tsdfg.inp_names[0]].dtype.as_numpy_dtype().type is np.float32
