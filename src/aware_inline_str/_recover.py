"""Function-boundary propagation: scope merging and return wrapping."""

from __future__ import annotations

import dataclasses
import sys
import types
from collections.abc import Mapping

from ._core import (
    InlineStr,
    _is_dataclass_instance,
    _iter_inline,
    _plain,
    _prepare,
    _wrap_str,
)


# ---------------------------------------------------------------------------
# function-boundary propagation (decorator and AST import hook)
# ---------------------------------------------------------------------------


def _merge_scope(scope) -> dict[str, object]:
    """Merge mappings visible in *scope*; drop genuinely ambiguous keys."""
    merged: dict[str, object] = {}
    conflicts: set[str] = set()
    for inline in _iter_inline(scope):
        for ch, seg in inline.mapping.items():
            if ch in merged and merged[ch] is not seg:
                conflicts.add(ch)
            else:
                merged.setdefault(ch, seg)
    for ch in conflicts:
        merged.pop(ch, None)
    return merged


def _wrap_value(value, merged: Mapping[str, object], seen: set[int] | None = None):
    """Wrap a return value, descending into exact built-in containers."""
    if isinstance(value, InlineStr):
        if not value.mapping and merged:
            return _wrap_str(_plain(value), merged, cls=type(value))
        return value
    if isinstance(value, str):
        return _wrap_str(value, merged)

    if seen is None:
        seen = set()
    oid = id(value)
    if oid in seen:
        return value

    # namedtuple: keep the concrete type and field order.
    if isinstance(value, tuple) and hasattr(type(value), "_fields"):
        seen.add(oid)
        items = tuple(_wrap_value(x, merged, seen) for x in value)
        if all(a is b for a, b in zip(value, items)):
            return value
        try:
            return type(value)(*items)
        except Exception:
            return value

    # Dataclasses are a common attribute container.  ``replace`` copies, so
    # the caller's object is never mutated.
    if _is_dataclass_instance(value):
        seen.add(oid)
        changes: dict[str, object] = {}
        for field in dataclasses.fields(value):
            old = getattr(value, field.name)
            new = _wrap_value(old, merged, seen)
            if new is not old:
                changes[field.name] = new
        if not changes:
            return value
        try:
            return dataclasses.replace(value, **changes)
        except Exception:
            return value

    # SimpleNamespace is effectively an attribute dict.
    if type(value) is types.SimpleNamespace:
        seen.add(oid)
        changes = {
            key: _wrap_value(item, merged, seen)
            for key, item in vars(value).items()
        }
        if all(changes[key] is item for key, item in vars(value).items()):
            return value
        return types.SimpleNamespace(**changes)

    if type(value) not in (list, tuple, dict, set, frozenset):
        return value

    seen.add(oid)

    if type(value) is list:
        items = [_wrap_value(x, merged, seen) for x in value]
        if all(a is b for a, b in zip(value, items)):
            return value
        return items
    if type(value) is tuple:
        items = tuple(_wrap_value(x, merged, seen) for x in value)
        if all(a is b for a, b in zip(value, items)):
            return value
        return items
    if type(value) is dict:
        items = {k: _wrap_value(v, merged, seen) for k, v in value.items()}
        if all(items[k] is v for k, v in value.items()):
            return value
        return items
    if type(value) is set or type(value) is frozenset:
        # Equality cannot tell wrapped from unwrapped here: an InlineStr is
        # equal to a plain str with the same text, so compare identity.
        old = list(value)
        new = [_wrap_value(x, merged, seen) for x in old]
        if all(a is b for a, b in zip(old, new)):
            return value
        return type(value)(new)
    return value


def scope_here():
    """Return the caller's locals (CPython-specific)."""
    return sys._getframe(1).f_locals


def wrap_return(value, scope):
    """Attach mappings from InlineStr values in *scope* to *value*."""
    merged = _merge_scope(scope)
    if not merged:
        return value
    return _wrap_value(value, merged)


def wrap_return_here(value):
    """``wrap_return`` using the caller's locals (CPython-specific)."""
    return wrap_return(value, sys._getframe(1).f_locals)


def wrap_iterable(iterable, scope):
    """Wrap every item produced by ``yield from``.

    This is a delegating generator: ``send``, ``throw`` and ``close`` are
    forwarded to the wrapped iterator, and its ``StopIteration.value`` is
    recovered as well.
    """
    gen = iter(iterable)
    if not hasattr(gen, "send"):
        for item in gen:
            yield wrap_return(item, scope)
        return

    try:
        item = next(gen)
    except StopIteration as exc:
        return wrap_return(exc.value, scope)

    while True:
        try:
            sent = yield wrap_return(item, scope)
        except GeneratorExit:
            close = getattr(gen, "close", None)
            if close is not None:
                close()
            raise
        except BaseException as exc:
            try:
                item = gen.throw(exc)
            except StopIteration as stop:
                return wrap_return(stop.value, scope)
        else:
            try:
                item = gen.send(sent)
            except StopIteration as stop:
                return wrap_return(stop.value, scope)


def _reserved_from_objects(objects) -> dict[str, object]:
    """Collect keys used by InlineStr values in *objects*.

    Keys that two different objects claim are dropped: they are genuinely
    ambiguous and no re-keying of an argument can repair them.
    """
    reserved: dict[str, object] = {}
    conflicts: set[str] = set()
    for inline in _iter_inline(objects):
        for ch, seg in inline.mapping.items():
            if ch in reserved and reserved[ch] is not seg:
                conflicts.add(ch)
            else:
                reserved.setdefault(ch, seg)
    for ch in conflicts:
        reserved.pop(ch, None)
    return reserved


def prepare_many(values, with_scope: bool = False):
    """Re-key conflicting top-level InlineStr values.

    Returns a replacement tuple only when something changed, so generated
    prologues are a no-op in the common case.  Nested containers are used for
    literal-collision detection but are not rebuilt.

    ``with_scope=True`` (used by the AST hook) also looks at the caller's
    locals and treats keys already used there as reserved.  This catches
    conflicts between an argument and a value that was not passed as an
    argument (a free variable or a local assigned by the caller).  The frame
    is only inspected when at least one top-level value is an InlineStr.
    """
    vals = list(values)
    reserved = None
    if with_scope and any(isinstance(v, InlineStr) for v in vals):
        try:
            scope = sys._getframe(1).f_locals
        except Exception:
            scope = None
        if scope:
            reserved = _reserved_from_objects(scope)
    new_vals, _merged = _prepare(
        vals,
        recursive=False,
        text_types=InlineStr._text_types,
        reserved=reserved,
    )
    if all(a is b for a, b in zip(vals, new_vals)):
        return None
    return tuple(new_vals)
