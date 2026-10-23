"""Configurable call adapters consulted by AST-rewritten call sites."""

from __future__ import annotations

import threading


# ---------------------------------------------------------------------------
# configurable call adapters
# ---------------------------------------------------------------------------
#
# The AST hook can rewrite ``obj.name(...)`` and plain ``f(...)`` calls to a
# runtime dispatcher.  The dispatcher consults registries of *adapters*, so
# the set of intercepted names is configuration, not something baked into the
# transformer.
#
# A method adapter is called as ``adapter(obj, name, args, kwargs)`` and
# returns either a replacement result or ``NOT_HANDLED`` to fall through.
# A call adapter is called as ``adapter(func, name, args, kwargs)``, where
# ``name`` is ``getattr(func, "__name__", None)``.  Registering under ``"*"``
# makes an adapter apply to every rewritten call/method.

DEFAULT_MATCH_METHODS = frozenset({"group", "groups", "groupdict"})


class _NotHandled:
    __slots__ = ()

    def __repr__(self) -> str:
        return "NOT_HANDLED"


NOT_HANDLED = _NotHandled()

_METHOD_ADAPTERS: dict[str, list] = {}
_CALL_ADAPTERS: dict[str, list] = {}
_ADAPTER_LOCK = threading.RLock()


def register_method_adapter(name: str, adapter) -> None:
    """Run *adapter* for instrumented ``obj.<name>(...)`` calls.

    ``name="*"`` matches every rewritten method call.  Adapters are tried in
    registration order; return ``NOT_HANDLED`` to continue with the next one.
    """
    with _ADAPTER_LOCK:
        _METHOD_ADAPTERS.setdefault(name, []).append(adapter)


def unregister_method_adapter(name: str, adapter=None) -> None:
    """Remove *adapter* (or every adapter) registered for *name*."""
    with _ADAPTER_LOCK:
        if adapter is None:
            _METHOD_ADAPTERS.pop(name, None)
            return
        adapters = _METHOD_ADAPTERS.get(name)
        if adapters and adapter in adapters:
            adapters.remove(adapter)
            if not adapters:
                del _METHOD_ADAPTERS[name]


def register_call_adapter(name: str, adapter) -> None:
    """Run *adapter* for instrumented plain ``f(...)`` calls named *name*.

    ``name="*"`` matches every rewritten call.  Adapters receive the callable
    itself as the first argument, so they can inspect it instead of relying on
    the name alone.
    """
    with _ADAPTER_LOCK:
        _CALL_ADAPTERS.setdefault(name, []).append(adapter)


def unregister_call_adapter(name: str, adapter=None) -> None:
    """Remove *adapter* (or every adapter) registered for *name*."""
    with _ADAPTER_LOCK:
        if adapter is None:
            _CALL_ADAPTERS.pop(name, None)
            return
        adapters = _CALL_ADAPTERS.get(name)
        if adapters and adapter in adapters:
            adapters.remove(adapter)
            if not adapters:
                del _CALL_ADAPTERS[name]


def _dispatch_method(obj, name, *args, **kwargs):
    """Runtime target of a rewritten ``obj.name(...)`` call."""
    adapters = _METHOD_ADAPTERS.get(name)
    if adapters:
        for adapter in tuple(adapters):
            result = adapter(obj, name, args, kwargs)
            if result is not NOT_HANDLED:
                return result
    adapters = _METHOD_ADAPTERS.get("*")
    if adapters:
        for adapter in tuple(adapters):
            result = adapter(obj, name, args, kwargs)
            if result is not NOT_HANDLED:
                return result
    return getattr(obj, name)(*args, **kwargs)


def _dispatch_call(func, *args, **kwargs):
    """Runtime target of a rewritten plain ``f(...)`` call."""
    name = getattr(func, "__name__", None)
    if name is not None:
        adapters = _CALL_ADAPTERS.get(name)
        if adapters:
            for adapter in tuple(adapters):
                result = adapter(func, name, args, kwargs)
                if result is not NOT_HANDLED:
                    return result
    adapters = _CALL_ADAPTERS.get("*")
    if adapters:
        for adapter in tuple(adapters):
            result = adapter(func, name, args, kwargs)
            if result is not NOT_HANDLED:
                return result
    return func(*args, **kwargs)
