# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Primitive translator for dot operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from dace.libraries import blas as dace_libblas

from jace import translator, util


if TYPE_CHECKING:
    import dace
    from jax._src import core as jax_core


@translator.register_primitive_translator()
@translator.make_primitive_translator("dot_general")
def dot_general_translator(
    builder: translator.JaxprTranslationBuilder,
    in_var_names: Sequence[str | None],
    out_var_names: Sequence[str],
    eqn: jax_core.JaxprEqn,
    eqn_state: dace.SDFGState,
) -> dace.SDFGState:
    """
    Implements the translation of the `dot_general` primitive.

    The `dot_general` primitive is a very general way to describe tensor operations,
    see: https://openxla.org/xla/operation_semantics#dotgeneral for more.
    The translator expands to a library node.

    Args:
        builder: The builder object of the translation.
        in_var_names: The SDFG variables used an input arguments. First is the name
            of the array containing the left hand side, the second name is the array
            containing the right hand side of the operation.
        out_var_names: Names of SDFG variables that should be used as outputs.
        eqn: The equation that should be translated.
        eqn_state: State into which the nested SDFG should be constructed.

    Notes:
        Currently no custom batch dimensions and contraction dimensions are supported,
        thus only regular matrix matrix/vector multiplication and dot products are
        supported.
    """
    assert len(eqn.outvars) == 1
    assert len(eqn.invars) == 2  # noqa: PLR2004 [magic-value-comparison]

    lhs_shape = util.get_jax_var_shape(eqn.invars[0])
    rhs_shape = util.get_jax_var_shape(eqn.invars[1])

    contraction_dims = eqn.params["dimension_numbers"][0]
    batch_dims = eqn.params["dimension_numbers"][1]

    # In JAX it is possible to specify an accumulator type and some library nodes
    #  in DaCe also support this.
    # TODO(phimuell): Check this.
    # TODO(phimuell): Patch DaCe to accept the accumulator type everywhere.
    accumulator_type = (
        None
        if eqn.params["preferred_element_type"] is None
        else util.translate_dtype(eqn.params["preferred_element_type"])
    )

    if batch_dims != ((), ()):
        raise NotImplementedError("'dot_general': Batching is not supported.")

    if (len(lhs_shape) == 1 and len(rhs_shape) == 1) and contraction_dims == ((0,), (0,)):
        assert lhs_shape[0] == rhs_shape[0]
        lhs, rhs = (eqn_state.add_read(name) for name in in_var_names)
        out = eqn_state.add_write(out_var_names[0])

        libnode = dace_libblas.Dot(
            f"_dot_general_DOT_{out_var_names[0]}",
            n=lhs_shape[0],
            accumulator_type=accumulator_type,
        )
        eqn_state.add_node(libnode)
        eqn_state.add_edge(lhs, None, libnode, "_x", builder.sdfg.make_array_memlet(lhs.data))
        eqn_state.add_edge(rhs, None, libnode, "_y", builder.sdfg.make_array_memlet(rhs.data))
        eqn_state.add_edge(libnode, "_result", out, None, builder.sdfg.make_array_memlet(out.data))

    elif (
        (len(lhs_shape) == 2) and (len(rhs_shape) in {2, 1}) and (contraction_dims == ((1,), (0,)))  # noqa: PLR2004 [magic-value-comparison]
    ):
        # Matrix-Matrix and Matrix-Vector Multiplication

        lhs, rhs = (eqn_state.add_read(name) for name in in_var_names)
        out = eqn_state.add_write(out_var_names[0])

        libnode = dace_libblas.MatMul(f"_dot_general_MatMul_{out_var_names[0]}")
        eqn_state.add_node(libnode)

        eqn_state.add_edge(lhs, None, libnode, "_a", builder.sdfg.make_array_memlet(lhs.data))
        eqn_state.add_edge(rhs, None, libnode, "_b", builder.sdfg.make_array_memlet(rhs.data))
        eqn_state.add_edge(libnode, "_c", out, None, builder.sdfg.make_array_memlet(out.data))

    else:
        # TODO(phimuell): Implement these cases with `TensorDot`.
        #   Tensordot does not have a batch dimension argument, however, if only
        #   one side has the batch dimension then we can emulate the behaviour
        #   using the permutation, if both have batch dimensions, then we have
        #   to create a loop that does it, possible copying a clot of stuff.
        raise NotImplementedError(
            "'dot_general': Non standard contracting and sizes are not supported."
        )
