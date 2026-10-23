"""The ``@inline_aware`` decorator."""

from __future__ import annotations

import functools
import inspect

from ._core import InlineStr, _prepare
from ._recover import _reserved_from_objects, wrap_return, wrap_iterable


# ---------------------------------------------------------------------------
# inline_aware
# ---------------------------------------------------------------------------


def _reserved_keys(func) -> dict[str, object]:
    """Keys used by captured/default values, minus genuinely ambiguous ones."""
    return _reserved_from_objects(
        (*_closure_values(func), *_function_defaults(func))
    )


def _prepare_decorator_args(func, args, kwargs):
    """Re-key arguments against each other *and* against captured values.

    Closure cells and defaults contribute their keys as *reserved*: an
    argument can never keep a key character already used by a captured
    InlineStr, while captured plain text is not mistaken for a literal (it
    is not necessarily part of the result).
    """
    vals, _merged = _prepare(
        [*args, *kwargs.values()],
        recursive=False,
        text_types=InlineStr._text_types,
        reserved=_reserved_keys(func),
    )
    n = len(args)
    return tuple(vals[:n]), dict(zip(kwargs, vals[n:]))


def _closure_values(func) -> tuple[object, ...]:
    """Return live closure-cell contents (for lambdas and nested functions)."""
    closure = getattr(func, "__closure__", None)
    if not closure:
        return ()
    out: list[object] = []
    for cell in closure:
        try:
            out.append(cell.cell_contents)
        except ValueError:
            pass
    return tuple(out)


def _function_defaults(func) -> tuple[object, ...]:
    defaults = getattr(func, "__defaults__", None) or ()
    kwdefaults = getattr(func, "__kwdefaults__", None) or {}
    return (*defaults, *kwdefaults.values())


def _call_scope(func, args, kwargs):
    return (
        *args,
        *kwargs.values(),
        *_closure_values(func),
        *_function_defaults(func),
    )


def inline_aware(func):
    """Decorate a function so arguments are prepared and results recovered.

    Works for normal, async, generator and async-generator functions, and is
    also used to instrument lambdas in the AST hook.  Closure values are
    included in the recovery scope, so nested functions and lambdas can
    recover mappings from captured InlineStr values too.
    """
    if inspect.isasyncgenfunction(func):
        @functools.wraps(func)
        async def async_gen_wrapper(*args, **kwargs):
            new_args, new_kwargs = _prepare_decorator_args(func, args, kwargs)
            scope = _call_scope(func, new_args, new_kwargs)
            agen = func(*new_args, **new_kwargs)
            try:
                item = await agen.__anext__()
            except StopAsyncIteration:
                return
            while True:
                try:
                    sent = yield wrap_return(item, scope)
                except GeneratorExit:
                    await agen.aclose()
                    raise
                except BaseException as exc:
                    try:
                        item = await agen.athrow(exc)
                    except StopAsyncIteration:
                        return
                else:
                    try:
                        item = await agen.asend(sent)
                    except StopAsyncIteration:
                        return

        return async_gen_wrapper

    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            new_args, new_kwargs = _prepare_decorator_args(func, args, kwargs)
            scope = _call_scope(func, new_args, new_kwargs)
            result = await func(*new_args, **new_kwargs)
            return wrap_return(result, scope)

        return async_wrapper

    if inspect.isgeneratorfunction(func):
        @functools.wraps(func)
        def gen_wrapper(*args, **kwargs):
            new_args, new_kwargs = _prepare_decorator_args(func, args, kwargs)
            scope = _call_scope(func, new_args, new_kwargs)
            # ``yield from`` preserves send/throw/close and the final
            # StopIteration.value (all recovered by wrap_iterable).
            return (yield from wrap_iterable(func(*new_args, **new_kwargs), scope))

        return gen_wrapper

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_args, new_kwargs = _prepare_decorator_args(func, args, kwargs)
        scope = _call_scope(func, new_args, new_kwargs)
        result = func(*new_args, **new_kwargs)
        return wrap_return(result, scope)

    return wrapper
