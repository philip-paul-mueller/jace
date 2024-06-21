# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Reimplementation of the `jax.stages` module.

This module reimplements the public classes of that JAX module.
However, because JaCe uses DaCe as backend they differ is some small aspects.

As in JAX JaCe has different stages, the terminology is taken from
[JAX' AOT-Tutorial](https://jax.readthedocs.io/en/latest/aot.html).
- Stage out:
    In this phase an executable Python function is translated to a Jaxpr.
- Lower:
    This will transform the Jaxpr into its SDFG equivalent.
- Compile:
    This will turn the SDFG into an executable object.
- Execution:
    This is the actual running of the computation.

As in JAX the in JaCe the user only has access to the last tree stages and
staging out and lowering is handled as a single step.
"""

from __future__ import annotations

import copy
import inspect
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, TypeVar, Union

from jax import tree_util as jax_tree

from jace import api, optimization, tracing, translated_jaxpr_sdfg as tjsdfg, translator, util
from jace.optimization import CompilerOptions
from jace.translator import pre_post_translation as pptrans
from jace.util import translation_cache as tcache


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import dace

__all__ = [
    "CompilerOptions",  # export for compatibility with JAX.
    "JaCeCompiled",
    "JaCeLowered",
    "JaCeWrapped",
    "Stage",
    "finalize_compilation_options",
    "get_active_compiler_options",
    "update_active_compiler_options",
]

#: Known compilation stages in JaCe.
Stage = Union["JaCeWrapped", "JaCeLowered", "JaCeCompiled"]

# These are used to annotated the `Stages`, however, there are some limitations.
#  First, the only stage that is fully annotated is `JaCeWrapped`. Second, since
#  static arguments modify the type signature of `JaCeCompiled.__call__()`, see
#  [JAX](https://jax.readthedocs.io/en/latest/aot.html#lowering-with-static-arguments)
#  for more, its argument can not be annotated, only its return type can.
#  However, in case of scalar return values, the return type is wrong anyway, since
#  JaCe and JAX for that matter, transforms scalars to arrays. Since there is no way of
#  changing that, but from a semantic point they behave the same so it should not
#  matter too much.
_P = ParamSpec("_P")
_ReturnType = TypeVar("_ReturnType")


class JaCeWrapped(tcache.CachingStage["JaCeLowered"], Generic[_P, _ReturnType]):
    """
    A function ready to be specialized, lowered, and compiled.

    This class represents the output of functions such as `jace.jit()` and is
    the first stage in the translation/compilation chain of JaCe. A user should
    never create a `JaCeWrapped` object directly, instead `jace.jit` should be
    used. While it supports just-in-time lowering and compilation, by just
    calling it, these steps can also be performed explicitly.
    The lowering, performed by this stage is cached, thus if a `JaCeWrapped`
    object is later lowered with the same arguments the result might be taken
    from the cache.

    Furthermore, a `JaCeWrapped` object is composable with all JAX transformations,
    all other stages are not.

    Args:
        fun: The function that is wrapped.
        primitive_translators: Primitive translators that that should be used.
        jit_options: Options to influence the jit process.

    Todo:
        - Support default values of the wrapped function.
        - Support static arguments.

    Note:
        The tracing of function will always happen with enabled `x64` mode,
        which is implicitly and temporary activated during tracing.
    """

    _fun: Callable[_P, _ReturnType]
    _primitive_translators: dict[str, translator.PrimitiveTranslator]
    _jit_options: api.JITOptions

    def __init__(
        self,
        fun: Callable[_P, _ReturnType],
        primitive_translators: Mapping[str, translator.PrimitiveTranslator],
        jit_options: api.JITOptions,
    ) -> None:
        assert all(
            param.default is param.empty for param in inspect.signature(fun).parameters.values()
        )
        super().__init__()
        self._primitive_translators = {**primitive_translators}
        self._jit_options = {**jit_options}
        self._fun = fun

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _ReturnType:
        """
        Executes the wrapped function, lowering and compiling as needed in one step.

        This function will lower and compile in one go. The function accepts the same
        arguments as the original computation and the return value is unflattened.

        Note:
            This function is also aware if a JAX tracing is going on. In this
            case, it will forward the computation.
            Currently, this function ignores the value of `jax.disable_jit()`.
        """
        if util.is_tracing_ongoing(*args, **kwargs):
            return self._fun(*args, **kwargs)

        lowered = self.lower(*args, **kwargs)
        compiled = lowered.compile()
        # TODO(phimuell): Filter out static arguments
        return compiled(*args, **kwargs)

    @tcache.cached_transition
    def lower(self, *args: _P.args, **kwargs: _P.kwargs) -> JaCeLowered[_ReturnType]:
        """
        Lower the wrapped computation for the given arguments.

        Performs the first two steps of the AOT steps described above, i.e. trace the
        wrapped function with the given arguments and stage it out to a Jaxpr. Then
        translate it to an SDFG. The result is encapsulated inside a `JaCeLowered`
        object that can later be compiled.

        It should be noted that the current lowering process will hard code the strides
        and the storage location of the input inside the SDFG. Thus if the SDFG is
        lowered with arrays in C order, calling the compiled SDFG with FORTRAN order
        will result in an error.

        Note:
            The tracing is always done with activated `x64` mode.
        """
        jaxpr_maker = tracing.make_jaxpr(
            fun=self._fun,
            trace_options=self._jit_options,
            return_out_tree=True,
        )
        jaxpr, out_tree = jaxpr_maker(*args, **kwargs)
        builder = translator.JaxprTranslationBuilder(
            primitive_translators=self._primitive_translators
        )
        trans_ctx: translator.TranslationContext = builder.translate_jaxpr(jaxpr)

        flat_call_args = jax_tree.tree_leaves((args, kwargs))
        tsdfg: tjsdfg.TranslatedJaxprSDFG = pptrans.postprocess_jaxpr_sdfg(
            trans_ctx=trans_ctx,
            fun=self.wrapped_fun,
            flat_call_args=flat_call_args,
        )

        # NOTE: `tsdfg` is deepcopied as a side effect of post processing.
        return JaCeLowered(tsdfg, out_tree)

    @property
    def wrapped_fun(self) -> Callable:
        """Return the underlying Python function."""
        return self._fun

    def _make_call_description(
        self, in_tree: jax_tree.PyTreeDef, flat_call_args: Sequence[Any]
    ) -> tcache.StageTransformationSpec:
        """
        Computes the key for the `JaCeWrapped.lower()` call inside the cache.

        For all non static arguments the function will generate an abstract description
        of an argument and for all static arguments the concrete value.

        Notes:
            The abstract description also includes storage location, i.e. if on CPU or
            on GPU, and the strides of the arrays.
        """
        # TODO(phimuell): Implement static arguments
        flat_call_args = tuple(tcache._AbstractCallArgument.from_value(x) for x in flat_call_args)
        return tcache.StageTransformationSpec(
            stage_id=id(self), flat_call_args=tuple(flat_call_args), in_tree=in_tree
        )


class JaCeLowered(tcache.CachingStage["JaCeCompiled"], Generic[_ReturnType]):
    """
    Represents the original computation as an SDFG.

    This class is the output type of `JaCeWrapped.lower()` and represents the original
    computation as an SDFG. This stage is followed by the `JaCeCompiled` stage, by
    calling `self.compile()`. A user should never directly construct a `JaCeLowered`
    object directly, instead `JaCeWrapped.lower()` should be used.

    The SDFG is optimized before the compilation, see `JaCeLowered.compile()` for how to
    control the process.

    Args:
        tsdfg: The lowered SDFG with metadata.
        out_tree: The pytree describing how to unflatten the output.

    Note:
        `self` will manage the passed `tsdfg` object. Modifying it results is undefined
        behavior. Although `JaCeWrapped` is composable with JAX transformations
        `JaCeLowered` is not.
    """

    _translated_sdfg: tjsdfg.TranslatedJaxprSDFG
    _out_tree: jax_tree.PyTreeDef

    def __init__(
        self,
        tsdfg: tjsdfg.TranslatedJaxprSDFG,
        out_tree: jax_tree.PyTreeDef,
    ) -> None:
        super().__init__()
        self._translated_sdfg = tsdfg
        self._out_tree = out_tree

    @tcache.cached_transition
    def compile(self, compiler_options: CompilerOptions | None = None) -> JaCeCompiled[_ReturnType]:
        """
        Optimize and compile the lowered SDFG using `compiler_options`.

        To perform the optimizations `jace_optimize()` is used. The actual options that
        are forwarded to it are obtained by passing `compiler_options` to
        `finalize_compilation_options()`.

        Args:
            compiler_options: The optimization options to use.
        """
        # We **must** deepcopy before we do any optimization, because all optimizations
        #  are in place, to properly cache stages, stages needs to be immutable.
        tsdfg: tjsdfg.TranslatedJaxprSDFG = copy.deepcopy(self._translated_sdfg)
        optimization.jace_optimize(tsdfg=tsdfg, **finalize_compilation_options(compiler_options))

        return JaCeCompiled(
            csdfg=tjsdfg.compile_jaxpr_sdfg(tsdfg),
            out_tree=self._out_tree,
        )

    def compiler_ir(self, dialect: str | None = None) -> tjsdfg.TranslatedJaxprSDFG:
        """
        Returns the internal SDFG.

        The function returns a `TranslatedJaxprSDFG` object. Direct modification of the
        returned object is forbidden and results in undefined behaviour.
        """
        if (dialect is None) or (dialect.upper() == "SDFG"):
            return self._translated_sdfg
        raise ValueError(f"Unknown dialect '{dialect}'.")

    def as_sdfg(self) -> dace.SDFG:
        """
        Returns the encapsulated SDFG.

        Modifying the returned SDFG in any way is undefined behavior.
        """
        return self.compiler_ir().sdfg

    def _make_call_description(
        self, in_tree: jax_tree.PyTreeDef, flat_call_args: Sequence[Any]
    ) -> tcache.StageTransformationSpec:
        """
        Creates the key for the `self.compile()` transition function.

        The key will depend on the final values that were used for optimization, i.e.
        they it will also include the global set of optimization options.
        """
        unflatted_args, unflatted_kwargs = jax_tree.tree_unflatten(in_tree, flat_call_args)
        assert (not unflatted_kwargs) and (len(unflatted_args) <= 1)

        options = finalize_compilation_options(unflatted_args[0] if unflatted_args else {})
        flat_options, option_tree = jax_tree.tree_flatten(options)
        return tcache.StageTransformationSpec(
            stage_id=id(self), flat_call_args=tuple(flat_options), in_tree=option_tree
        )


class JaCeCompiled(Generic[_ReturnType]):
    """
    Compiled version of the SDFG.

    This is the last stage of the JaCe's jit chain. A user should never create a
    `JaCeCompiled` instance, instead `JaCeLowered.compile()` should be used.

    Since the strides and storage location of the arguments, that where used to lower
    the computation are hard coded inside the SDFG, a `JaCeCompiled` object can only be
    called with compatible arguments.

    Args:
        csdfg: The compiled SDFG object.
        inp_names: SDFG variables used as inputs.
        out_names: SDFG variables used as outputs.
        out_tree: Pytree describing how to unflatten the output.

    Note:
        The class assumes ownership of its input arguments.

    Todo:
        - Automatic strides adaptation.
    """

    _csdfg: tjsdfg.CompiledJaxprSDFG
    _out_tree: jax_tree.PyTreeDef

    def __init__(
        self,
        csdfg: tjsdfg.CompiledJaxprSDFG,
        out_tree: jax_tree.PyTreeDef,
    ) -> None:
        self._csdfg = csdfg
        self._out_tree = out_tree

    def __call__(self, *args: Any, **kwargs: Any) -> _ReturnType:
        """
        Calls the embedded computation.

        Note:
            Unlike the `lower()` function which takes the same arguments as the original
            computation, to call this function you have to remove all static arguments.
            Furthermore, all arguments must have strides and storage locations that is
            compatible with the ones that were used for lowering.
        """
        flat_call_args = jax_tree.tree_leaves((args, kwargs))
        flat_output = self._csdfg(flat_call_args)
        if flat_output is None:
            return None  # type: ignore[return-value]  # Type confusion.
        return jax_tree.tree_unflatten(self._out_tree, flat_output)


# <--------------------------- Compilation/Optimization options management

_JACELOWERED_ACTIVE_COMPILE_OPTIONS: CompilerOptions = optimization.DEFAULT_OPTIMIZATIONS.copy()
"""Global set of currently active compilation/optimization options.

The global set is initialized with `jace.optimization.DEFAULT_OPTIMIZATIONS`. It can be
managed through `update_active_compiler_options()` and accessed through
`get_active_compiler_options()`, however, it is advised that a user should use
`finalize_compilation_options()` for getting the final options that should be used
for optimization.
"""


def update_active_compiler_options(new_active_options: CompilerOptions) -> CompilerOptions:
    """
    Updates the set of active compiler options.

    Merges the options passed as `new_active_options` with the currently active
    compiler options. This set is used by `JaCeLowered.compile()` to determine
    which options should be used.
    The function will return the set of options that was active before the call.

    To obtain the set of currently active options use `get_active_compiler_options()`.

    Todo:
        Make a proper context manager.
    """
    previous_active_options = _JACELOWERED_ACTIVE_COMPILE_OPTIONS.copy()
    _JACELOWERED_ACTIVE_COMPILE_OPTIONS.update(new_active_options)
    return previous_active_options


def get_active_compiler_options() -> CompilerOptions:
    """Returns the set of currently active compiler options."""
    return _JACELOWERED_ACTIVE_COMPILE_OPTIONS.copy()


def finalize_compilation_options(compiler_options: CompilerOptions | None) -> CompilerOptions:
    """
    Returns the final compilation options.

    There are two different sources of optimization options. The first one is the global
    set of currently active compiler options. The second one is the options that are
    passed to this function, which takes precedence. Thus, the `compiler_options`
    argument describes the difference from the currently active global options.

    This function is used by `JaCeLowered` if it has to determine which options to use
    for optimization, either for compiling the lowered SDFG or for computing the key.

    Args:
        compiler_options: The local compilation options.

    See Also:
        `get_active_compiler_options()` to inspect the set of currently active options
        and `update_active_compiler_options()` to modify them.
    """
    return get_active_compiler_options() | (compiler_options or {})
