# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements the translator related to data movement."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from jace import translator
from jace.translator import mapped_operation_base_translator as mapped_base


if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax import core as jax_core


class CopyTranslator(mapped_base.MappedOperationTranslatorBase):
    """Copy operations are implemented as a map to ensure that they can be fused with other maps."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(primitive_name="copy")

    @override
    def write_tasklet_code(
        self,
        tskl_ranges: Sequence[tuple[str, str]],
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> str:
        return "__out = __in0"


class DevicePutTranslator(mapped_base.MappedOperationTranslatorBase):
    """The `device_put` primitive is used to transfer data between host and device.

    The current implementation only supports the copying where the data already is. Currently DaCe
    only knows about the Host and the GPU. Furthermore, currently JaCe works in such a way that
    everything is either put on the host or the device. Because of this, the `DevicePutTranslator`
    is, currently, just a simple copy operation that should be removed, by the optimization.

    Todo:
        - Make into a Memlet because only the Memlet can handle copying between devices.
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(primitive_name="device_put")

    @override
    def write_tasklet_code(
        self,
        tskl_ranges: Sequence[tuple[str, str]],
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> str:
        if not (eqn.params["device"] is None and eqn.params["src"] is None):
            raise NotImplementedError(
                f"Can only copy on the host, but not from {eqn.params['src']} to {eqn.params['device']}."
            )
        return "__out = __in0"


_ = translator.register_primitive_translator(CopyTranslator())
_ = translator.register_primitive_translator(DevicePutTranslator())
