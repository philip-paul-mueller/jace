# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Implements the tracing machinery that is used to build the Jaxpr.

JAX provides `jax.make_jaxpr()`, which is essentially a debug utility, but it does not
provide any other public way to get a Jaxpr. This module provides the necessary
functionality for this in JaCe.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar, overload

import jax
from jax import core as jax_core, tree_util as jax_tree


if TYPE_CHECKING:
    from collections.abc import Callable

    from jace import api

_P = ParamSpec("_P")
_ReturnType = TypeVar("_ReturnType")


@overload
def make_jaxpr(
    fun: Callable[_P, _ReturnType],
    trace_options: api.JITOptions,
    return_out_tree: Literal[True],
) -> Callable[_P, tuple[jax_core.ClosedJaxpr, jax_tree.PyTreeDef]]: ...


@overload
def make_jaxpr(
    fun: Callable[_P, _ReturnType],
    trace_options: api.JITOptions,
    return_out_tree: Literal[False] = False,
) -> Callable[_P, jax_core.ClosedJaxpr]: ...


def make_jaxpr(
    fun: Callable[_P, Any],
    trace_options: api.JITOptions,
    return_out_tree: bool = False,
) -> Callable[_P, tuple[jax_core.ClosedJaxpr, jax_tree.PyTreeDef] | jax_core.ClosedJaxpr]:
    """
    JaCe's replacement for `jax.make_jaxpr()`.

    Returns a callable object that produces a Jaxpr and optionally a pytree defining
    the output. By default the callable will only return the Jaxpr, however, by setting
    `return_out_tree` the function will also return the output tree, this is different
    from the `return_shape` of `jax.make_jaxpr()`.

    Currently the tracing is always performed with an enabled `x64` mode.

    Returns:
        The function returns a callable that will perform the tracing on the passed
        arguments. If `return_out_tree` is `False` that callable will simply return the
        generated Jaxpr. If `return_out_tree` is `True` the function will return a tuple
        with the Jaxpr and a pytree object describing the structure of the output.

    Args:
        fun: The original Python computation.
        trace_options: The options used for tracing, the same arguments that
            are supported by `jace.jit`.
        return_out_tree: Also return the pytree of the output.

    Todo:
        - Handle default arguments of `fun`.
        - Handle static arguments.
        - Turn `trace_options` into a `TypedDict` and sync with `jace.jit`.
    """
    if trace_options:
        raise NotImplementedError(
            f"Not supported tracing options: {', '.join(f'{k}' for k in trace_options)}"
        )
    assert all(param.default is param.empty for param in inspect.signature(fun).parameters.values())

    def tracer_impl(
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> tuple[jax_core.ClosedJaxpr, jax_tree.PyTreeDef] | jax_core.ClosedJaxpr:
        # In JAX `float32` is the main datatype, and they go to great lengths to avoid
        #  some aggressive [type promotion](https://jax.readthedocs.io/en/latest/type_promotion.html).
        #  However, in this case we will have problems when we call the SDFG, for some
        #  reasons `CompiledSDFG` does not work in that case correctly, thus we enable
        #  it for the tracing.
        with jax.experimental.enable_x64():
            # TODO(phimuell): copy the implementation of the real tracing
            jaxpr_maker = jax.make_jaxpr(
                fun,
                **trace_options,
                return_shape=True,
            )
            jaxpr, out_shapes = jaxpr_maker(
                *args,
                **kwargs,
            )

        if not return_out_tree:
            return jaxpr

        # Regardless what the documentation of `make_jaxpr` claims, it does not output
        #  a pytree but an abstract description of the shape, that we will
        #  transform into a pytree.
        out_tree = jax_tree.tree_structure(out_shapes)
        return jaxpr, out_tree

    return tracer_impl
