# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Reimplementation of the `jax.stages` module.

This module reimplements the public classes of that Jax module.
However, they are a big different, because JaCe uses DaCe as backend.

As in Jax JaCe has different stages, the terminology is taken from
[Jax' AOT-Tutorial](https://jax.readthedocs.io/en/latest/aot.html).
- Stage out:
    In this phase we translate an executable python function into Jaxpr.
- Lower:
    This will transform the Jaxpr into an SDFG equivalent. As a implementation note,
    currently this and the previous step are handled as a single step.
- Compile:
    This will turn the SDFG into an executable object, see `dace.codegen.CompiledSDFG`.
- Execution:
    This is the actual running of the computation.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import jax as _jax

from jace import optimization, translator, util
from jace.optimization import CompilerOptions
from jace.translator import post_translation as ptrans
from jace.util import dace_helper, translation_cache as tcache


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import dace


class JaCeWrapped(tcache.CachingStage["JaCeLowered"]):
    """A function ready to be specialized, lowered, and compiled.

    This class represents the output of functions such as `jace.jit()` and is the first stage in
    the translation/compilation chain of JaCe. A user should never create a `JaCeWrapped` object
    directly, instead `jace.jit` should be used for that.
    While it supports just-in-time lowering and compilation these steps can also be performed
    explicitly. The lowering performed by this stage is cached, thus if a `JaCeWrapped` object is
    lowered later, with the same argument the result is taken from the cache.
    Furthermore, a `JaCeWrapped` object is composable with all Jax transformations.

    Args:
        fun:                    The function that is wrapped.
        primitive_translators:  The list of primitive translators that that should be used.
        jit_options:            Options to influence the jit process.

    Todo:
        - Handle pytrees.
        - Handle all options to `jax.jit`.

    Note:
        The tracing of function will always happen with enabled `x64` mode, which is implicitly
        and temporary activated during tracing.
    """

    _fun: Callable
    _primitive_translators: dict[str, translator.PrimitiveTranslator]
    _jit_options: dict[str, Any]

    def __init__(
        self,
        fun: Callable,
        primitive_translators: Mapping[str, translator.PrimitiveTranslator],
        jit_options: Mapping[str, Any],
    ) -> None:
        super().__init__()
        # We have to shallow copy both the translator and the jit options.
        #  This prevents that any modifications affect `self`.
        #  Shallow is enough since the translators themselves are immutable.
        self._primitive_translators = dict(primitive_translators)
        self._jit_options = dict(jit_options)
        self._fun = fun

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Executes the wrapped function, lowering and compiling as needed in one step."""

        # If we are inside a traced context, then we forward the call to the wrapped function.
        #  This ensures that JaCe is composable with Jax.
        if util.is_tracing_ongoing(*args, **kwargs):
            return self._fun(*args, **kwargs)

        lowered = self.lower(*args, **kwargs)
        compiled = lowered.compile()
        return compiled(*args, **kwargs)

    @tcache.cached_transition
    def lower(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> JaCeLowered:
        """Lower this function explicitly for the given arguments.

        Performs the first two steps of the AOT steps described above, i.e. stage out to Jaxpr
        and then translate to SDFG. The result is encapsulated and returned into a `Lowered` object.
        """
        if len(kwargs) != 0:
            raise NotImplementedError("Currently only positional arguments are supported.")

        # TODO(phimuell): Currently the SDFG that we build only supports `C_CONTIGUOUS` memory
        #  order. Since we support the paradigm that "everything passed to `lower` should also
        #  be accepted as argument to call the result", we forbid other memory orders here.
        if not all((not util.is_array(arg)) or arg.flags["C_CONTIGUOUS"] for arg in args):
            raise NotImplementedError("Currently can not handle strides beside 'C_CONTIGUOUS'.")

        # In Jax `float32` is the main datatype, and they go to great lengths to avoid some
        #  aggressive [type promotion](https://jax.readthedocs.io/en/latest/type_promotion.html).
        #  However, in this case we will have problems when we call the SDFG, for some reasons
        #  `CompiledSDFG` does not work in that case correctly, thus we enable it for the tracing.
        with _jax.experimental.enable_x64():
            driver = translator.JaxprTranslationDriver(
                primitive_translators=self._primitive_translators
            )
            jaxpr = _jax.make_jaxpr(self._fun)(*args)
            trans_ctx: translator.TranslationContext = driver.translate_jaxpr(jaxpr)

        # Perform the post processing and turn it into a `TranslatedJaxprSDFG` that can be
        #  compiled and called later.
        # NOTE: `tsdfg` was deepcopied as a side effect of post processing.
        tsdfg: translator.TranslatedJaxprSDFG = ptrans.postprocess_jaxpr_sdfg(
            trans_ctx=trans_ctx,
            fun=self.wrapped_fun,
        )

        return JaCeLowered(tsdfg)

    @property
    def wrapped_fun(self) -> Callable:
        """Returns the wrapped function."""
        return self._fun

    def _make_call_description(
        self,
        *args: Any,
    ) -> tcache.StageTransformationSpec:
        """This function computes the key for the `JaCeWrapped.lower()` call to cache it.

        The function will compute a full abstract description on its argument. Currently it is
        only able to handle positional argument and does not support static arguments.
        """
        call_args = tuple(tcache._AbstractCallArgument.from_value(x) for x in args)
        return tcache.StageTransformationSpec(stage_id=id(self), call_args=call_args)


class JaCeLowered(tcache.CachingStage["JaCeCompiled"]):
    """Represents the original computation as an SDFG.

    It represents the computation wrapped by a `JaCeWrapped` translated and lowered to SDFG.
    It is followed by the `JaCeCompiled` stage.
    Although, `JaCeWrapped` is composable with Jax transformations `JaCeLowered` is not.
    A user should never create such an object, instead `JaCeWrapped.lower()` should be used.

    Note:
        `self` will manage the passed `tsdfg` object. Modifying it results in undefined behavior.

    Todo:
        - Handle pytrees.
    """

    _translated_sdfg: translator.TranslatedJaxprSDFG

    def __init__(
        self,
        tsdfg: translator.TranslatedJaxprSDFG,
    ) -> None:
        super().__init__()
        self._translated_sdfg = tsdfg

    @tcache.cached_transition
    def compile(
        self,
        compiler_options: CompilerOptions | None = None,
    ) -> JaCeCompiled:
        """Optimize and compile the lowered SDFG using `compiler_options`.

        Returns an object that encapsulates a compiled SDFG object. To influence the various
        optimizations and compile options of JaCe you can use the `compiler_options` argument.
        If nothing is specified `jace.optimization.DEFAULT_OPTIMIZATIONS` will be used.

        Note:
            Before `compiler_options` is forwarded to `jace_optimize()` it will be merged with
            the default arguments.
        """
        # We **must** deepcopy before we do any optimization, because all optimizations are in
        #  place, however, to properly cache stages, they have to be immutable.
        tsdfg: translator.TranslatedJaxprSDFG = copy.deepcopy(self._translated_sdfg)
        optimization.jace_optimize(tsdfg=tsdfg, **self._make_compiler_options(compiler_options))

        return JaCeCompiled(
            csdfg=util.compile_jax_sdfg(tsdfg),
            inp_names=tsdfg.inp_names,
            out_names=tsdfg.out_names,
        )

    def compiler_ir(self, dialect: str | None = None) -> translator.TranslatedJaxprSDFG:
        """Returns the internal SDFG.

        The function returns a `TranslatedJaxprSDFG` object.
        It is important that modifying this object in any way is undefined behavior.
        """
        if (dialect is None) or (dialect.upper() == "SDFG"):
            return self._translated_sdfg
        raise ValueError(f"Unknown dialect '{dialect}'.")

    def as_html(self, filename: str | None = None) -> None:
        """Runs the `view()` method of the underlying SDFG."""
        self.compiler_ir().sdfg.view(filename=filename, verbose=False)

    def as_sdfg(self) -> dace.SDFG:
        """Returns the encapsulated SDFG.

        Modifying the returned SDFG in any way is undefined behavior.
        """
        return self.compiler_ir().sdfg

    def _make_call_description(
        self,
        compiler_options: CompilerOptions | None = None,
    ) -> tcache.StageTransformationSpec:
        """This function computes the key for the `self.compile()` call to cache it.

        The key that is computed by this function is based on the concrete values of the passed
        compiler options. This is different from the key computed by `JaCeWrapped` which is an
        abstract description.
        """
        options = self._make_compiler_options(compiler_options)
        call_args = tuple(sorted(options.items(), key=lambda X: X[0]))
        return tcache.StageTransformationSpec(stage_id=id(self), call_args=call_args)

    def _make_compiler_options(
        self,
        compiler_options: CompilerOptions | None,
    ) -> CompilerOptions:
        return optimization.DEFAULT_OPTIMIZATIONS | (compiler_options or {})


class JaCeCompiled:
    """Compiled version of the SDFG.

    This is the last stage of the jit chain. A user should never create a `JaCeCompiled` instance,
    instead `JaCeLowered.compile()` should be used.

    Args:
        csdfg:      The compiled SDFG object.
        inp_names:  Names of the SDFG variables used as inputs.
        out_names:  Names of the SDFG variables used as outputs.

    Note:
        The class assumes ownership of its input arguments.

    Todo:
        - Handle pytrees.
    """

    _csdfg: dace_helper.CompiledSDFG  # The compiled SDFG object.
    _inp_names: tuple[str, ...]  # Name of all input arguments.
    _out_names: tuple[str, ...]  # Name of all output arguments.

    def __init__(
        self,
        csdfg: dace_helper.CompiledSDFG,
        inp_names: Sequence[str],
        out_names: Sequence[str],
    ) -> None:
        if (not inp_names) or (not out_names):
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
            args,
            kwargs,
        )


#: Known compilation stages in JaCe.
Stage = JaCeWrapped | JaCeLowered | JaCeCompiled


__all__ = [
    "CompilerOptions",  # export for compatibility with Jax.
    "JaCeCompiled",
    "JaCeLowered",
    "JaCeWrapped",
    "Stage",
]
