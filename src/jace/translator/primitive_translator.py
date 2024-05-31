# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Interface for all primitive translators and managing of the global translator registry.

Todo:
    Implement proper context manager for working with the registry.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Literal, Protocol, cast, overload, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, MutableMapping, MutableSequence, Sequence

    import dace
    from jax import core as jax_core

    from jace import translator

#: Global registry of the active primitive translators.
#:  The `dict` maps the name of a primitive to its associated translators.
_PRIMITIVE_TRANSLATORS_REGISTRY: dict[str, translator.PrimitiveTranslator] = {}


class PrimitiveTranslatorCallable(Protocol):
    """Callable version of the primitive translators.

    Used for type annotation purposes, classes should be derived from `PrimitiveTranslator` instead.
    You can use `jace.translator.make_primitive_translator()` to add a `primitive` property to
    a callable.
    """

    @abstractmethod
    def __call__(
        self,
        builder: translator.JaxprTranslationBuilder,
        in_var_names: Sequence[str | None],
        out_var_names: MutableSequence[str],
        eqn: jax_core.JaxprEqn,
        eqn_state: dace.SDFGState,
    ) -> dace.SDFGState | None:
        """Translates the Jax primitive into its SDFG equivalent.

        Before the builder calls this function it will perform the following preparatory tasks:
        - It will allocate the SDFG variables that are used as outputs. Their names will be passed
            through the `out_var_names` argument, in the same order as `eqn.outvars`.
        - It will collect the names of the SDFG variables that are used as inputs and place them in
            `in_var_names`, in the same order as `eqn.invars`. If an input argument refers to a
            literal no SDFG variable is created for it and `None` is used to indicate this.
        - The builder will create variables that are used as output. They are passed as
            `out_var_names`, same order as in the equation.
        - The builder will create a new terminal state and pass it as `eqn_state` argument. This
            state is guaranteed to be empty and `translator.terminal_sdfg_state is eqn_state` holds.

        Then the primitive translator is called.
        Usually a primitive translator should construct the dataflow graph inside `eqn_state`.
        However, it is allowed that the primitive translators creates more states if needed, but
        this state machinery has to have a single terminal state, which must be returned and
        reachable from `eqn_state`. If the function returns `None` the builder will assume that
        primitive translator was able to fully construct the dataflow graph within `eqn_state`.

        While a primitive translator is forbidden from meddling with the input variables mentioned
        in `in_var_names` in any way, it is allowed to modify the output variables. For example
        a translator could create a new SDFG variable, with different strides. But in that case
        the primitive translator must update the internal mapping of the builder TBA HOW, and
        modify the names passed through `out_var_names`. However, the translator is allowed to
        create internal temporary variables without registering them to the mapping, as long as it
        uses the supplied variables as final output. To ensure that there are no collision with
        further variables, the translator should prefix them.

        Args:
            builder:        The builder object of the translation.
            in_var_names:   List of the names of the arrays created inside the
                                SDFG for the inpts or `None` in case of a literal.
            out_var_names:  List of the names of the arrays created inside the
                                SDFG for the outputs.
            eqn:            The Jax primitive that should be translated.
            eqn_state:      State into which the primitive`s SDFG representation
                                should be constructed.
        """
        ...


@runtime_checkable
class PrimitiveTranslator(PrimitiveTranslatorCallable, Protocol):
    """Interface for all Jax primitive translators.

    A translator for a primitive translates a single equation of a Jaxpr into its SDFG equivalent.
    For satisfying this interface a concrete implementation must be immutable after construction.

    Primitive translators are simple, but highly specialized objects that are only able to perform
    the translation of a single primitive. The overall translation process itself is managed by a
    builder object, which also owns and manage the primitive translators. In the end this implements
    the delegation pattern.

    You can use `jace.translator.register_primitive_translator()` to register your translator to JaCe.
    """

    @property
    @abstractmethod
    def primitive(self) -> str:
        """Returns the name of the Jax primitive that `self` is able to handle."""
        ...


@overload
def make_primitive_translator(
    primitive: str,
    primitive_translator: Literal[None] = None,
) -> Callable[[translator.PrimitiveTranslatorCallable], translator.PrimitiveTranslator]: ...


@overload
def make_primitive_translator(
    primitive: str, primitive_translator: translator.PrimitiveTranslatorCallable
) -> translator.PrimitiveTranslator: ...


def make_primitive_translator(
    primitive: str,
    primitive_translator: translator.PrimitiveTranslatorCallable | None = None,
) -> (
    Callable[[translator.PrimitiveTranslatorCallable], translator.PrimitiveTranslator]
    | translator.PrimitiveTranslator
):
    """Turn `primitive_translator` into a `PrimitiveTranslator` for primitive `primitive`.

    Essentially, this function adds the `primitive` property to a callable, such that it satisfy
    the `PrimitiveTranslator` protocol. However, it does not add it to the registry, for that
    `register_primitive_translator()` has to be used.

    Notes:
        This function cal also be used as decorator.
    """

    def wrapper(
        primitive_translator: translator.PrimitiveTranslatorCallable,
    ) -> translator.PrimitiveTranslator:
        if getattr(primitive_translator, "primitive", primitive) != primitive:
            raise ValueError(
                f"Tried to change the 'primitive' property of '{primitive_translator}' from "
                f"'{primitive_translator.primitive}' to '{primitive}'."  # type: ignore[attr-defined]
            )
        primitive_translator.primitive = primitive  # type: ignore[attr-defined]  # We define the attribute.
        return cast("translator.PrimitiveTranslator", primitive_translator)

    return wrapper if primitive_translator is None else wrapper(primitive_translator)


@overload
def register_primitive_translator(
    primitive_translator: Literal[None] = None,
    overwrite: bool = False,
) -> Callable[[translator.PrimitiveTranslator], translator.PrimitiveTranslator]: ...


@overload
def register_primitive_translator(
    primitive_translator: translator.PrimitiveTranslator,
    overwrite: bool = False,
) -> translator.PrimitiveTranslator: ...


def register_primitive_translator(
    primitive_translator: translator.PrimitiveTranslator | None = None,
    overwrite: bool = False,
) -> (
    translator.PrimitiveTranslator
    | Callable[[translator.PrimitiveTranslator], translator.PrimitiveTranslator]
):
    """Adds a primitive translator to JaCe's global registry.

    The default set of primitives that are used if nothing is specified to to `jace.jit` are stored
    inside a global registry. To add a translator to this registry this function can be used.

    If a translator for `primitive` is already registered an error will be generated. However,
    by specifying `overwrite` `primitive_translator` will replace the current one.

    Args:
        primitive_translator: The primitive translator to add to the global registry.
        overwrite:            Replace the current primitive translator with `primitive_translator`.

    Note:
        To add a `primitive` property use the `@make_primitive_translator` decorator.
        This function returns `primitive_translator` unmodified, which allows it to be
        used as decorator.
    """

    def wrapper(
        primitive_translator: translator.PrimitiveTranslator,
    ) -> translator.PrimitiveTranslator:
        if primitive_translator.primitive in _PRIMITIVE_TRANSLATORS_REGISTRY and not overwrite:
            raise ValueError(
                f"Explicit override=True needed for primitive '{primitive_translator.primitive}' "
                "to overwrite existing one."
            )
        _PRIMITIVE_TRANSLATORS_REGISTRY[primitive_translator.primitive] = primitive_translator
        return primitive_translator

    return wrapper if primitive_translator is None else wrapper(primitive_translator)


def get_regsitered_primitive_translators() -> dict[str, translator.PrimitiveTranslator]:
    """Returns a copy of the current state of JaCe's global primitive registry.

    The state returned by this function is compatible to what `jace.hit`'s `primitive_translators`
    argument expects. It is important the the returned object is decoupled from the registry.
    """
    return _PRIMITIVE_TRANSLATORS_REGISTRY.copy()


def set_active_primitive_translators_to(
    new_translators: Mapping[str, translator.PrimitiveTranslator],
) -> MutableMapping[str, translator.PrimitiveTranslator]:
    """Exchange the global translator registry state of JaCe with `new_translators`.

    The function will return the state of the global translator registry prior to this call.
    Any changes to `new_translators` after calling this function will have no effect on the
    global translator registry and vice versa.
    """
    global _PRIMITIVE_TRANSLATORS_REGISTRY
    assert all(getattr(trans, "primitive", prim) for prim, trans in new_translators.items())
    previous_translators = _PRIMITIVE_TRANSLATORS_REGISTRY
    _PRIMITIVE_TRANSLATORS_REGISTRY = dict(new_translators)
    return previous_translators
