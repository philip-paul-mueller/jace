# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This module contains the functionality related to the compilation cache of the stages.

Actually there are two different caches:
- The lowering cache.
- And the compilation cache.

Both are implemented as a singleton.
"""

from __future__ import annotations

import functools as ft
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import dace
from jax import core as jax_core

from jace import util
from jace.jax import stages


def cached_translation(
    action: Callable,
) -> Callable:
    """Decorator for making the transfer method, i.e. `JaceWrapped.lower()` and `JaceLowered.compile()` cacheable.

    The cache is global and the function will add the respecifve cache object to the object upon its first call.
    To clear the caches use the `clear_translation_cache()` function.
    """

    @ft.wraps(action)
    def _action_wrapper(
        self: stages.Stage,
        *args: Any,
        **kwargs: Any,
    ) -> stages.Stage:
        if hasattr(self, "_cache"):
            cache: TranslationCache = self._cache
        else:
            cache = _get_cache(self)
            self._cache = cache
        key: _CachedCall = cache.make_key(self, *args, **kwargs)
        if cache.has(key):
            return cache.get(key)
        next_stage: stages.Stage = action(self, *args, **kwargs)
        cache.add(key, next_stage)
        return next_stage

    return _action_wrapper


def clear_translation_cache() -> None:
    """Clear all caches associated to translation."""

    if not hasattr(_get_cache, "_caches"):
        return
    _get_cache._caches.clear()
    return


def _get_cache(
    self: stages.Stage,
    size: int = 128,
) -> TranslationCache:
    """Returns the cache associated to `name`.

    If called for the first time, the cache sizes will be set to `size`.
    In all later calls this value is ignored.
    """
    # Get the caches and if not present, create them.
    if not hasattr(_get_cache, "_caches"):
        _caches: dict[type[stages.Stage], TranslationCache] = {}
        _get_cache._caches = _caches  # type: ignore[attr-defined]  # ruff removes the `getattr()` calls
    _caches = _get_cache._caches  # type: ignore[attr-defined]

    if type(self) not in _caches:
        _caches[type(self)] = TranslationCache(size=size)

    return _caches[type(self)]


@dataclass(init=True, eq=True, frozen=True)
class _AbstarctCallArgument:
    """Class to represent one argument to the call in an abstract way.

    It is used as part of the key in the cache.
    It represents the structure of the argument, i.e. its shape, type and so on, but nots its value.
    To construct it you should use the `from_value()` class function which interfere the characteristics from a value.
    """

    shape: tuple[int, ...] | tuple[()]
    dtype: dace.typeclass
    strides: tuple[int, ...] | tuple[()] | None
    storage: dace.StorageType

    @classmethod
    def from_value(
        cls,
        val: Any,
    ) -> _AbstarctCallArgument:
        """Construct an `_AbstarctCallArgument` from a value.

        Todo:
            Improve, such that NumPy arrays are on CPU, CuPy on GPU and so on.
            This function also probably fails for scalars.
        """
        if not util.is_fully_addressable(val):
            raise NotImplementedError("Distributed arrays are not addressed yet.")
        if isinstance(val, jax_core.Literal):
            raise TypeError("Jax Literals are not supported as cache keys.")

        if util.is_array(val):
            if util.is_jax_array(val):
                val = val.__array__(copy=False)
            shape = val.shape
            dtype = util.translate_dtype(val.dtype)
            strides = getattr(val, "strides", None)
            # TODO(phimuell): is `CPU_Heap` always okay? There would also be `CPU_Pinned`.
            storage = (
                dace.StorageType.GPU_Global if util.is_on_device(val) else dace.StorageType.CPU_Heap
            )

            return cls(shape=shape, dtype=dtype, strides=strides, storage=storage)

        if isinstance(val, jax_core.ShpedArray):
            shape = val.aval.shape
            dtype = val.aval.dtype
            strides = None
            storage = (
                dace.StorageType.GPU_Global
                if util.is_on_device(val.val)
                else dace.StorageType.CPU_Heap
            )

            return cls(shape=shape, dtype=dtype, strides=strides, storage=storage)

        if isinstance(val, jax_core.AbstractValue):
            raise TypeError(f"Can not make 'JaCeVar' from '{type(val).__name__}', too abstract.")

        # If we are here, then we where not able, thus we will will now try Jax
        #  This is inefficient and we should make it better.
        return cls.from_value(jax_core.get_aval(val))


@runtime_checkable
class _ConcreteCallArgument(Protocol):
    """Type for encoding a concrete arguments in the cache."""

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass


@dataclass(init=True, eq=True, frozen=True)
class _CachedCall:
    """Represents the structure of the entire call in the cache.

    This class represents both the `JaceWrapped.lower()` and `JaceLowered.compile()` call.
    The key combines the "origin of the call", i.e. `self` and the call arguments.

    Arguments are represented in two ways:
    - `_AbstarctCallArgument`: Which encode only the structure of the arguments.
        These are essentially the tracer used by Jax.
    - `_ConcreteCallArgument`: Which represents actual values of the call.
        These are either the static arguments or compile options.

    Depending of the origin the call, the key used for caching is different.
    For `JaceWrapped` only the wrapped callable is included in the cache.

    For the `JaceLowered` the SDFG is used as key, however, in a very special way.
    `dace.SDFG` does not define `__hash__()` or `__eq__()` thus these operations fall back to `object`.
    However, an SDFG defines the `hash_sdfg()` function, which generates a hash based on the structure of the SDFG.
    We use the SDFG because we want to cache on it, but since it is not immutable, we have to account for that, by including this structural hash.
    This is not ideal but it should work in the beginning.
    """

    fun: Callable | None
    sdfg: dace.SDFG | None
    sdfg_hash: int | None
    fargs: tuple[
        _AbstarctCallArgument
        | _ConcreteCallArgument
        | tuple[str, _AbstarctCallArgument]
        | tuple[str, _ConcreteCallArgument],
        ...,
    ]

    @classmethod
    def make_key(
        cls,
        stage: stages.Stage,
        *args: Any,
        **kwargs: Any,
    ) -> _CachedCall:
        """Creates a cache key for the stage object `stage` that was called to advance to the next stage."""

        if isinstance(stage, stages.JaceWrapped):
            # JaceWrapped.lower() to JaceLowered
            #   Currently we only allow positional arguments and no static arguments.
            #   Thus the function argument part of the key only consists of abstract arguments.
            if len(kwargs) != 0:
                raise NotImplementedError("'kwargs' are not implemented in 'JaceWrapped.lower()'.")
            fun = stage.__wrapped__
            sdfg = None
            sdfg_hash = None
            fargs: tuple[_AbstarctCallArgument, ...] = tuple(
                _AbstarctCallArgument.from_value(x) for x in args
            )

        elif isinstance(stage, stages.JaceLowered):
            # JaceLowered.compile() to JaceCompiled
            #   We only accepts compiler options, which the Jax interface mandates
            #   are inside a `dict` thus we will get at most one argument.
            fun = None
            sdfg = stage.compiler_ir().sdfg
            sdfg_hash = int(sdfg.hash_sdfg(), 16)

            if len(kwargs) != 0:
                raise ValueError(
                    "All arguments to 'JaceLowered.compile()' must be inside a 'dict'."
                )
            if len(args) >= 2:
                raise ValueError("Only a 'dict' is allowed as argument to 'JaceLowered.compile()'.")
            if (len(args) == 0) or (args[0] is None):
                # No compiler options where specified, so we use the default ones.
                comp_ops: stages.CompilerOptions = stages.JaceLowered.DEF_COMPILER_OPTIONS
            else:
                # Compiler options where given.
                comp_ops = args[0]
            assert isinstance(comp_ops, dict)
            assert all(
                isinstance(k, str) and isinstance(v, _ConcreteCallArgument)
                for k, v in comp_ops.items()
            )

            # We will now make `(argname, argvalue)` pairs and sort them according to `argname`.
            #  This guarantees a stable order.
            fargs: tuple[tuple[str, _ConcreteCallArgument], ...] = tuple(  # type: ignore[no-redef]  # Type confusion.
                sorted(
                    ((argname, argvalue) for argname, argvalue in comp_ops.items()),
                    key=lambda X: X[0],
                )
            )

        else:
            raise TypeError(f"Can not make key from '{type(stage).__name__}'.")

        return cls(fun=fun, sdfg=sdfg, sdfg_hash=sdfg_hash, fargs=fargs)


class TranslationCache:
    """The _internal_ cache object.

    It implements a simple LRU cache, for storing the results of the `JaceWrapped.lower()` and `JaceLowered.compile()` calls.
    You should not use this cache directly but instead use the `cached_translation` decorator.

    Notes:
        The most recently used entry is at the end of the `OrderedDict`.
            The reason for this is, because there the new entries are added.
    """

    __slots__ = ["_memory", "_size"]

    _memory: OrderedDict[_CachedCall, stages.Stage]
    _size: int

    def __init__(
        self,
        size: int = 128,
    ) -> None:
        """Creates a cache instance of size `size`."""
        if size <= 0:
            raise ValueError(f"Invalid cache size of '{size}'")
        self._memory: OrderedDict[_CachedCall, stages.Stage] = OrderedDict()
        self._size = size

    @staticmethod
    def make_key(
        stage: stages.Stage,
        *args: Any,
        **kwargs: Any,
    ) -> _CachedCall:
        """Create a key object for `stage`."""
        return _CachedCall.make_key(stage, *args, **kwargs)

    def has(
        self,
        key: _CachedCall,
    ) -> bool:
        """Check if `self` have a record of `key`.

        Notes:
            For generating `key` use the `make_key()` function.
            This function will not modify the order of the cached entries.
        """
        return key in self._memory

    def get(
        self,
        key: _CachedCall,
    ) -> stages.Stage:
        """Get the next stage associated with `key`.

        Notes:
            It is an error if `key` does not exist.
            This function will mark `key` as most recently used.
        """
        if not self.has(key):
            raise KeyError(f"Key '{key}' is unknown.")
        self._memory.move_to_end(key, last=True)
        return self._memory.get(key)  # type: ignore[return-value]  # type confusion

    def add(
        self,
        key: _CachedCall,
        res: stages.Stage,
    ) -> TranslationCache:
        """Adds `res` under `key` to `self`.

        Notes:
            It is not an error if if `key` is already present.
        """
        if self.has(key):
            # `key` is known, so move it to the end and update the mapped value.
            self._memory.move_to_end(key, last=True)
            self._memory[key] = res

        else:
            # `key` is not known so we have to add it
            while len(self._memory) >= self._size:
                self._evict(None)
            self._memory[key] = res
        return self

    def _evict(
        self,
        key: _CachedCall | None,
    ) -> bool:
        """Evict `key` from `self` and return `True`.

        In case `key` is not known the function returns `False`.
        If `key` is `None` then evict the oldest one unconditionally.
        """
        if key is None:
            if len(self._memory) == 0:
                return False
            self._memory.popitem(last=False)
            return True

        if not self.has(key):
            return False
        self._memory.move_to_end(key, last=False)
        self._memory.popitem(last=False)
        return True