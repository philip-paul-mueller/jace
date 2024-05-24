# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import MutableSequence, Sequence

import dace
from jax import core as jax_core
from typing_extensions import override

from jace import translator


class ReshapeTranslator(translator.PrimitiveTranslator):
    """Reshapes an array.

    Todo:
        - Handle `dimensions` parameter fully.
        - Find a way to make it as a Map.
    """

    __slots__ = ()

    @property
    def primitive(self) -> str:
        return "reshape"

    @override
    def __call__(
        self,
        driver: translator.JaxprTranslationDriver,
        in_var_names: Sequence[str | None],
        out_var_names: MutableSequence[str],
        eqn: jax_core.JaxprEqn,
        eqn_state: dace.SDFGState,
    ) -> None:
        """Performs the reshaping.

        Currently a copy using a Memlet is performed.
        """
        if eqn.params["dimensions"] is not None:
            raise NotImplementedError("Currently 'dimensions' must be 'None'.")
        eqn_state.add_nedge(
            eqn_state.add_read(in_var_names[0]),
            eqn_state.add_write(out_var_names[0]),
            dace.Memlet(
                data=in_var_names[0],
                subset=", ".join(f"0:{size}" for size in eqn.invars[0].aval.shape),
                other_subset=", ".join(f"0:{size}" for size in eqn.params["new_sizes"]),
            ),
        )


translator.register_primitive_translator(ReshapeTranslator())
