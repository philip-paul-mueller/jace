# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of the `jace.jax.stages.Compiled` stage for Jace."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from jace import util
from jace.jax import stages
from jace.util import dace_helper as jdace


class JaceCompiled(stages.Compiled):
    """Compiled version of the SDFG.

    Todo:
        Handle pytrees.
    """

    __slots__ = (
        "_csdfg",
        "_inp_names",
        "_out_names",
    )

    _csdfg: jdace.CompiledSDFG  # The compiled SDFG object.
    _inp_names: tuple[str, ...]  # Name of all input arguments.
    _out_names: tuple[str, ...]  # Name of all output arguments.

    def __init__(
        self,
        csdfg: jdace.CompiledSDFG,
        inp_names: Sequence[str],
        out_names: Sequence[str],
    ) -> None:
        if (len(inp_names) == 0) or (len(out_names) == 0):
            raise ValueError("Input and output can not be empty.")
        self._csdfg = csdfg
        self._inp_names = tuple(inp_names)
        self._out_names = tuple(out_names)

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Calls the embedded computation."""
        return util.run_jax_sdfg(
            self._csdfg,
            self._inp_names,
            self._out_names,
            *args,
            **kwargs,
        )