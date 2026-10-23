"""A standalone, per-instance ``InlineStr`` with automatic propagation.

The class stores non-text message segments as private-use / unassigned
code points and keeps a ``char -> object`` mapping on *each instance*.
Unlike a ContextVar-based implementation, mappings therefore survive
re-entrancy, threads, async tasks and repeated execution of the same command.

What is covered automatically
-----------------------------
* All string operators: ``+``, ``*``, ``%``, indexing, slicing, iteration.
* Every normal ``str`` method that returns ``str`` / ``list[str]`` /
  ``tuple[str, ...]`` (``upper``, ``replace``, ``split``, ``strip``,
  ``format``, ``join``, ...).
* Conflict resolution: if two InlineStr values use the same key character for
  different objects, one side is re-keyed with a fresh PUA character.
  Two layers cooperate:
  - at construction, a live-key registry detects that a char is already used
    by another object and re-keys immediately (this also covers conflicts
    created *inside* a function, where entry-time preparation cannot see
    them); register with ``configure(key_registry=False)`` to opt out;
  - at function entry, arguments are re-keyed against each other and against
    values in the caller's locals (the AST hook) or the closure/defaults
    (``inline_aware``).
* Literal collisions: if a key character also occurs in a plain-string
  argument, the InlineStr side is re-keyed so the plain text is not mistaken
  for a segment.
* Optional AST import hook (``install_ast_hook``): function entry re-keys
  arguments; every ``return``, ``yield`` and ``yield from`` lifts a plain
  ``str`` result back to ``InlineStr`` using the mappings visible in local
  variables.  Results that are lists, tuples, dicts, sets, namedtuples,
  dataclasses or ``SimpleNamespace`` are wrapped recursively.
* Lambda bodies are wrapped with ``inline_aware`` by the same hook, and the
  decorator also covers closures and default arguments (``lambda s: s + "!"``).
* Generators keep their full protocol: ``send``, ``throw``, ``close`` and the
  ``StopIteration.value`` of ``yield from`` are forwarded and recovered.
  Async generators keep ``asend`` / ``athrow`` / ``aclose``.
* Attributes: dataclasses, namedtuples and ``SimpleNamespace`` are handled by
  default; ``InlineStr.configure(scan_objects=True)`` additionally walks
  ``vars(obj)`` for ordinary objects.
* Untransformed modules: ``patch_module(mod)`` wraps already-imported
  functions, methods, properties, staticmethods and classmethods at the
  Python call boundary (no source rewriting needed).
* ``re.Match``: ``m.group(n)`` is a C-level copy that loses the mapping.
  The AST hook rewrites ``.group`` / ``.groups`` / ``.groupdict`` calls inside
  instrumented code, so they return InlineStr slices directly -- no function
  boundary needed.  This is not special-cased in the transformer: the names
  come from a configurable adapter registry
  (``install_ast_hook(..., rewrite_methods=...)``, ``register_method_adapter``).
  Any other API can be adapted the same way, and ``rewrite_calls=True`` routes
  *every* call through the runtime dispatcher for blanket hooking.
  Uninstrumented code can additionally use ``inline_match(m)``,
  ``inline_group``, ``inline_groups``, ``inline_finditer`` or
  ``inline_findall`` -- a group is a contiguous slice of the source, so
  slicing the original InlineStr restores its mapping exactly.
* Optional ``@inline_aware`` decorator for selected functions without the
  hook (normal, async, generator and async-generator functions).

What cannot be covered
----------------------
Literal "all cases" is impossible for a ``str`` subclass, because a value can
leave the object graph through a C API and come back as a plain ``str``.
Concretely:

* ``"".join([...])`` with a *plain* separator, ``re.sub``, ``json.dumps``,
  ``os.path.join`` and any other C/Python function return plain ``str``.
  The AST return wrapper recovers them at a function boundary; without the
  hook the mapping is lost inside the function.
* ``str(s)`` / ``f"{a}{b}"`` / ``"{}".format(s)`` are handled by overriding
  ``__str__`` / ``__format__`` / ``format`` where Python dispatches to the
  subclass, but multi-field f-strings and ``str.format`` on a *plain* format
  string are compiled/implemented in C and can return plain ``str`` (the
  enclosing function's return/yield boundary recovers them).
* Transformations that change the key character itself (``translate`` to a
  non-key char, ``encode``, base64, compression, encryption,
  ``json.dumps(ensure_ascii=True)``, regex replacement of PUA chars) destroy
  provenance.
* Persistence (database, cache, file) loses the mapping unless pickled with
  this class (``pickle`` support is included).
* Code in modules that are not instrumented and that performs the C-level
  operation internally.  Install the AST hook before importing such modules,
  or use ``@inline_aware`` / ``patch_module`` at their public boundary.
* ``exec`` / ``eval`` are not instrumented; code compiled that way behaves
  like an untransformed module.
* If a plain string happens to contain the same PUA char as a key, the
  ambiguity is only resolvable when both sides are visible to ``_prepare``
  (operators/method arguments).  A value that returns from arbitrary C code
  is matched heuristically.
* Conflicting keys that escape both prevention layers (for example a value
  produced by another library that picked the same PUA char) are genuinely
  ambiguous at a plain-``str`` boundary.  The boundary recovery drops that
  key rather than misattributing it; the adapter helpers (``inline_group``
  etc.) and explicitly prepared call sites avoid the situation.

Usage
-----
::

    from aware_inline_str import InlineStr

    class Seg:
        def __init__(self, name): self.name = name
        def __repr__(self): return f"Seg({self.name!r})"

    s = InlineStr.from_iterable(["hello ", Seg("at"), " world"])
    print(s.to_list())          # ['hello ', Seg('at'), ' world']
    print((s + "!").to_list())  # ['hello ', Seg('at'), ' world!']

For automatic recovery across arbitrary functions, install the hook before
importing the target modules::

    from aware_inline_str import InlineStr, install_ast_hook
    install_ast_hook(__name__, "my_package")
    import my_package.mod

Or decorate selected functions::

    @inline_aware
    def render(user, seg):
        return re.sub("x", "y", "".join([user, seg]))
"""

from __future__ import annotations

import ast
import copy as _copy
import dataclasses
import functools
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import itertools
import random
import re
import sys
import threading
import types
import weakref
from collections.abc import Mapping

__all__ = [
    "InlineStr",
    "CodePointConflictError",
    "InlineMatch",
    "DEFAULT_MATCH_METHODS",
    "NOT_HANDLED",
    "register_call_adapter",
    "register_method_adapter",
    "unregister_call_adapter",
    "unregister_method_adapter",
    "inline_aware",
    "inline_findall",
    "inline_finditer",
    "inline_group",
    "inline_groups",
    "inline_match",
    "install_ast_hook",
    "patch_module",
    "prepare_many",
    "scope_here",
    "wrap_iterable",
    "wrap_return",
    "wrap_return_here",
]


class CodePointConflictError(ValueError):
    """Raised when no free private-use code point can be allocated."""


# ---------------------------------------------------------------------------
# code-point allocation
# ---------------------------------------------------------------------------

# BMP private-use area plus the two supplementary private-use areas.
# 6400 + 65534 + 65534 = 137468 code points.  They are unassigned by Unicode,
# stable across versions and safe to round-trip through UTF-8/UTF-16/UTF-32.
_PUA_RANGES = ((0xE000, 0xF8FF), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD))
_PUA_CPS = [cp for _lo, _hi in _PUA_RANGES for cp in range(_lo, _hi + 1)]

# Opt-in: also walk ``vars(obj)`` for ordinary objects in a recovery scope.
# Off by default because a bot object can reference a very large graph.
_SCAN_OBJECT_ATTRIBUTES = False

# Sequential allocation from a random start: two values created by this
# module in the same process never get the same key until all 137468
# private-use code points have been handed out.  Random allocation would
# collide with a birthday-probability that becomes significant in a
# long-running process.
_ALLOCATOR = itertools.count(random.randrange(len(_PUA_CPS)))

# Live key ownership: char -> weak set of InlineStr instances using it.
# Construction re-keys a mapping whose char is already used by a live
# instance for a *different* object, so conflicts created inside a function
# (not visible to entry-time preparation) never reach the return boundary.
_KEY_OWNERS: dict[str, "weakref.WeakSet[InlineStr]"] = {}
_KEY_LOCK = threading.RLock()


def _gen_char(used: set[str]) -> str:
    """Return a globally fresh private-use character (and add it to *used*)."""
    size = len(_PUA_CPS)
    start = next(_ALLOCATOR)
    for offset in range(size):
        ch = chr(_PUA_CPS[(start + offset) % size])
        if ch not in used:
            used.add(ch)
            return ch
    raise CodePointConflictError("no free private-use code point")


_MISSING = object()
_REGISTER_SWEEP = 0


def _rekey_locked(
    text: str, mapping: dict[str, object]
) -> tuple[str, dict[str, object]]:
    """Re-key chars already owned by a live value for a different object."""
    for ch in list(mapping):
        if ch not in text:
            continue
        owners = _KEY_OWNERS.get(ch)
        if owners is None:
            continue
        if not owners:
            # All previous owners died; drop the stale entry.
            del _KEY_OWNERS[ch]
            continue
        seg = mapping[ch]
        conflict = False
        for inst in list(owners):
            other = inst.mapping.get(ch, _MISSING)
            if other is not _MISSING and other is not seg:
                conflict = True
                break
        if not conflict:
            continue
        used = set(_KEY_OWNERS) | set(mapping) | set(text)
        new = _gen_char(used)
        text = text.replace(ch, new)
        mapping[new] = mapping.pop(ch)
    return text, mapping


def _register_locked(obj) -> None:
    """Register *obj* as a live owner of each key char present in its text."""
    global _REGISTER_SWEEP
    mapping = obj.mapping
    if not mapping:
        return
    text = _plain(obj)
    for ch in mapping:
        if ch not in text:
            continue
        owners = _KEY_OWNERS.get(ch)
        if owners is None:
            try:
                _KEY_OWNERS[ch] = weakref.WeakSet([obj])
            except TypeError:
                return  # subclass not weak-referenceable: skip tracking
        else:
            try:
                owners.add(obj)
            except TypeError:
                return
    _REGISTER_SWEEP += 1
    if _REGISTER_SWEEP >= 4096:
        _REGISTER_SWEEP = 0
        for ch in [ch for ch, owners in _KEY_OWNERS.items() if not owners]:
            del _KEY_OWNERS[ch]


def _plain(value: str) -> str:
    """Return an exact ``str`` for *value*, bypassing ``InlineStr.__str__``."""
    return str.__str__(value)


def _normalize_text_types(text_types) -> tuple[type, ...]:
    if text_types is None:
        return ()
    if isinstance(text_types, type):
        return (text_types,)
    return tuple(text_types)


def _text_of(value, text_types: tuple[type, ...] = ()) -> str | None:
    """Return the text of a string-like value, or ``None`` if not text-like."""
    if isinstance(value, InlineStr):
        return _plain(value)
    if isinstance(value, str):
        return value
    if text_types and isinstance(value, text_types):
        text = getattr(value, "text", None)
        if isinstance(text, str):
            return text
        if text is not None:
            return str(text)
    return None


# ---------------------------------------------------------------------------
# traversal helpers
# ---------------------------------------------------------------------------


def _iter_inline(obj, seen: set[int] | None = None):
    """Yield InlineStr values, descending into common containers."""
    if isinstance(obj, InlineStr):
        yield obj
        return
    if isinstance(obj, str) or obj is None:
        return
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return
    if isinstance(obj, (tuple, list, set, frozenset)):
        seen.add(oid)
        for item in obj:
            yield from _iter_inline(item, seen)
    elif isinstance(obj, Mapping):
        seen.add(oid)
        for item in obj.values():
            yield from _iter_inline(item, seen)
    elif _is_dataclass_instance(obj):
        # Dataclasses are a common way to return "containers of attributes".
        seen.add(oid)
        for field in dataclasses.fields(obj):
            yield from _iter_inline(getattr(obj, field.name), seen)
    elif type(obj) is types.SimpleNamespace:
        seen.add(oid)
        for item in vars(obj).values():
            yield from _iter_inline(item, seen)
    elif _SCAN_OBJECT_ATTRIBUTES and _is_plain_object(obj):
        seen.add(oid)
        try:
            values = vars(obj).values()
        except TypeError:
            return
        for item in values:
            yield from _iter_inline(item, seen)


def _is_plain_object(obj) -> bool:
    if isinstance(
        obj,
        (
            type,
            types.ModuleType,
            types.FunctionType,
            types.MethodType,
            types.BuiltinFunctionType,
        ),
    ):
        return False
    return hasattr(obj, "__dict__")


def _is_dataclass_instance(obj) -> bool:
    return (
        not isinstance(obj, type)
        and hasattr(type(obj), "__dataclass_fields__")
        and dataclasses.is_dataclass(obj)
    )


def _collect_literal(
    obj,
    literal: set[str],
    text_types: tuple[type, ...],
    seen: set[int],
) -> None:
    """Collect characters that are ordinary text, not mapping keys."""
    if isinstance(obj, InlineStr):
        for ch in _plain(obj):
            if ch not in obj.mapping:
                literal.add(ch)
        return
    if isinstance(obj, str):
        literal.update(obj)
        return
    if text_types and isinstance(obj, text_types):
        text = _text_of(obj, text_types)
        if text:
            literal.update(text)
        return
    oid = id(obj)
    if oid in seen:
        return
    if isinstance(obj, (tuple, list, set, frozenset)):
        seen.add(oid)
        for item in obj:
            _collect_literal(item, literal, text_types, seen)
    elif isinstance(obj, Mapping):
        seen.add(oid)
        for item in obj.values():
            _collect_literal(item, literal, text_types, seen)
    elif _is_dataclass_instance(obj):
        seen.add(oid)
        for field in dataclasses.fields(obj):
            _collect_literal(
                getattr(obj, field.name), literal, text_types, seen
            )
    elif type(obj) is types.SimpleNamespace:
        seen.add(oid)
        for item in vars(obj).values():
            _collect_literal(item, literal, text_types, seen)
    elif _SCAN_OBJECT_ATTRIBUTES and _is_plain_object(obj):
        seen.add(oid)
        try:
            values = vars(obj).values()
        except TypeError:
            return
        for item in values:
            _collect_literal(item, literal, text_types, seen)


def _prepare(
    values,
    *,
    recursive: bool = False,
    text_types: tuple[type, ...] = (),
    reserved: Mapping[str, object] | None = None,
) -> tuple[list, dict[str, object]]:
    """Make a set of values safe to combine.

    Every InlineStr mapping is walked.  Keys that conflict with an earlier
    mapping, or that also occur as ordinary text, are replaced (in the
    InlineStr text and mapping) by a fresh PUA character.

    ``reserved`` registers mapping keys that are already in use but whose
    values are not part of *values* (for example the closure cells of a
    decorated function).  Arguments are re-keyed away from them.

    ``recursive=True`` rebuilds list/tuple/dict/set arguments so that nested
    InlineStr values can be re-keyed too.  It is used for method arguments
    (``format``, ``%``, ``replace``, ...), where copying a container is
    harmless.  ``recursive=False`` only replaces top-level InlineStr values;
    nested values still contribute to the merged mapping, but a nested
    conflict cannot be repaired.

    Returns ``(new_values, merged_mapping)``.
    """
    values = list(values)

    top_ids = {id(v) for v in values if isinstance(v, InlineStr)}

    # Process top-level values first so they win conflicts against values
    # discovered inside containers.
    order: list[InlineStr] = []
    seen: set[int] = set()
    for v in values:
        if isinstance(v, InlineStr) and id(v) not in seen:
            seen.add(id(v))
            order.append(v)

    nested: list[InlineStr] = []

    def collect_nested(obj) -> None:
        if isinstance(obj, InlineStr):
            oid = id(obj)
            if oid not in seen:
                seen.add(oid)
                nested.append(obj)
            return
        if isinstance(obj, str) or obj is None:
            return
        oid = id(obj)
        if oid in seen:
            return
        if isinstance(obj, (tuple, list, set, frozenset)):
            seen.add(oid)
            for item in obj:
                collect_nested(item)
        elif isinstance(obj, Mapping):
            seen.add(oid)
            for item in obj.values():
                collect_nested(item)
        elif _is_dataclass_instance(obj):
            seen.add(oid)
            for field in dataclasses.fields(obj):
                collect_nested(getattr(obj, field.name))
        elif type(obj) is types.SimpleNamespace:
            seen.add(oid)
            for item in vars(obj).values():
                collect_nested(item)
        elif _SCAN_OBJECT_ATTRIBUTES and _is_plain_object(obj):
            seen.add(oid)
            try:
                values = vars(obj).values()
            except TypeError:
                pass
            else:
                for item in values:
                    collect_nested(item)

    for v in values:
        collect_nested(v)
    order.extend(nested)

    literal: set[str] = set()
    lit_seen: set[int] = set()
    for v in values:
        _collect_literal(v, literal, text_types, lit_seen)

    prepared: dict[int, InlineStr] = {}
    reserved_map: dict[str, object] = dict(reserved) if reserved else {}
    merged: dict[str, object] = dict(reserved_map)
    owner: dict[str, int] = {}

    for obj in order:
        oid = id(obj)
        text = _plain(obj)
        mapping = dict(obj.mapping)
        can_rekey = recursive or oid in top_ids
        changed = False

        for ch in list(mapping):
            seg = mapping[ch]
            # A shared character is only a problem when it denotes a
            # *different* object: two values agreeing on char -> object can
            # both keep it.
            conflict = ch in owner and merged.get(ch) is not seg
            reserved_conflict = (
                ch not in owner
                and ch in reserved_map
                and reserved_map[ch] is not seg
            )
            collision = ch in literal
            if conflict or reserved_conflict or collision:
                if can_rekey:
                    used = (
                        set(literal)
                        | set(owner)
                        | set(merged)
                        | set(mapping)
                        | set(text)
                    )
                    new = _gen_char(used)
                    text = text.replace(ch, new)
                    del mapping[ch]
                    mapping[new] = seg
                    if owner.get(ch) == oid:
                        del owner[ch]
                    owner.setdefault(new, oid)
                    changed = True
                    continue
                # Cannot repair a nested value without rebuilding its
                # container.  Keep the first mapping; the merged result is
                # best-effort in this case.
            owner.setdefault(ch, oid)

        new_obj = InlineStr(text, mapping) if changed else obj
        prepared[oid] = new_obj
        for ch, seg in mapping.items():
            merged.setdefault(ch, seg)

    if recursive:
        def rebuild(obj):
            if isinstance(obj, InlineStr):
                return prepared[id(obj)]
            if isinstance(obj, list):
                return [rebuild(x) for x in obj]
            if isinstance(obj, tuple):
                items = tuple(rebuild(x) for x in obj)
                if hasattr(type(obj), "_fields"):
                    try:
                        return type(obj)(*items)
                    except Exception:
                        return items
                return items
            if isinstance(obj, dict):
                return {k: rebuild(v) for k, v in obj.items()}
            if isinstance(obj, set):
                return {rebuild(x) for x in obj}
            if isinstance(obj, frozenset):
                return frozenset(rebuild(x) for x in obj)
            if _is_dataclass_instance(obj):
                changes = {
                    field.name: rebuild(getattr(obj, field.name))
                    for field in dataclasses.fields(obj)
                }
                if all(
                    changes[field.name] is getattr(obj, field.name)
                    for field in dataclasses.fields(obj)
                ):
                    return obj
                try:
                    return dataclasses.replace(obj, **changes)
                except Exception:
                    return obj
            if type(obj) is types.SimpleNamespace:
                changes = {k: rebuild(v) for k, v in vars(obj).items()}
                if all(
                    changes[k] is v for k, v in vars(obj).items()
                ):
                    return obj
                return types.SimpleNamespace(**changes)
            return obj

        new_values = [rebuild(v) for v in values]
    else:
        new_values = [
            prepared.get(id(v), v) if isinstance(v, InlineStr) else v
            for v in values
        ]

    return new_values, merged


def _wrap_str(
    text: str,
    mapping: Mapping[str, object],
    *,
    cls: type["InlineStr"] | None = None,
) -> "InlineStr":
    """Attach the compatible part of *mapping* to *text*."""
    cls = cls or InlineStr
    plain = _plain(text)
    filtered = {ch: seg for ch, seg in mapping.items() if ch in plain}
    if not filtered:
        return cls(plain)
    indices = {i: filtered[ch] for i, ch in enumerate(plain) if ch in filtered}
    return cls(plain, filtered, indices)


def _wrap_result(
    obj,
    mapping: Mapping[str, object],
    *,
    cls: type["InlineStr"] | None = None,
):
    """Wrap str / list[str] / tuple[str, ...] return values."""
    if isinstance(obj, str):
        return _wrap_str(obj, mapping, cls=cls)
    if isinstance(obj, list):
        return [_wrap_result(x, mapping, cls=cls) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_wrap_result(x, mapping, cls=cls) for x in obj)
    return obj


# ---------------------------------------------------------------------------
# InlineStr
# ---------------------------------------------------------------------------


class InlineStr(str):
    """A ``str`` carrying a per-instance ``char -> object`` mapping.

    The text contains one private-use character for each non-text object.
    ``mapping`` maps those characters to the original objects.
    ``inline_indices`` optionally maps exact positions to objects; it is
    maintained by all built-in operations and makes ``to_list`` precise even
    when a key character is repeated.

    While ``_key_registry`` is enabled (the default), construction checks a
    process-wide registry of live key chars and re-keys this instance if a
    char is already used for a *different* object.  This makes conflicts
    essentially impossible without explicit coordination, at the cost of a
    few hundred nanoseconds per construction; disable it with
    ``configure(key_registry=False)``.
    """

    _text_types: tuple[type, ...] = ()
    _error_type: type[Exception] = CodePointConflictError
    _key_registry: bool = True

    def __new__(
        cls,
        value: str = "",
        mapping: Mapping[str, object] | None = None,
        indices: Mapping[int, object] | None = None,
    ):
        # Support ``InlineStr(b"...", "utf-8")`` like ``str`` does.  This is
        # deliberately limited: use ``bytes.decode()`` for anything unusual.
        if isinstance(value, (bytes, bytearray, memoryview)):
            encoding = mapping if isinstance(mapping, str) else "utf-8"
            obj = super().__new__(cls, value, encoding)
            obj.mapping = {}
            obj.inline_indices = None
            return obj

        if mapping is None and isinstance(value, InlineStr):
            mapping = value.mapping
            if indices is None:
                indices = value.inline_indices

        if not mapping:
            obj = super().__new__(cls, value)
            obj.mapping = {}
            obj.inline_indices = dict(indices) if indices is not None else None
            return obj

        text = value if isinstance(value, str) else str(value)
        with _KEY_LOCK:
            mapping = dict(mapping)
            if cls._key_registry:
                text, mapping = _rekey_locked(text, mapping)
            obj = super().__new__(cls, text)
            obj.mapping = mapping
            obj.inline_indices = dict(indices) if indices is not None else None
            if cls._key_registry:
                _register_locked(obj)
        return obj

    def __init__(self, *args, **kwargs):
        # ``str`` is immutable; everything happens in ``__new__``.
        pass

    def __class_getitem__(cls, item):
        # ``InlineStr[str]`` in annotations is accepted for convenience.
        return cls

    # -- configuration ----------------------------------------------------

    @classmethod
    def configure(
        cls,
        *,
        text_types=None,
        memory_level=None,
        error_type=None,
        scan_objects=None,
        key_registry=None,
    ):
        """Configure text-like types, allocation errors and object scanning.

        ``text_types`` lists wrapper types whose ``.text`` attribute should be
        treated as text; it *replaces* the current tuple rather than adding
        to it.  ``memory_level`` is accepted for API compatibility and
        ignored.
        ``scan_objects=True`` makes recovery walk ``vars(obj)`` for ordinary
        objects in a function's locals/arguments (useful for attribute
        containers; potentially expensive for large object graphs).
        ``key_registry=False`` disables construction-time conflict checks (a
        small per-construction cost, see the class docstring).
        """
        global _SCAN_OBJECT_ATTRIBUTES
        if text_types is not None:
            cls._text_types = _normalize_text_types(text_types)
        if error_type is not None:
            cls._error_type = error_type
        if scan_objects is not None:
            _SCAN_OBJECT_ATTRIBUTES = bool(scan_objects)
        if key_registry is not None:
            cls._key_registry = bool(key_registry)
        return cls

    @classmethod
    def clear_map(cls):
        """Compatibility no-op: mappings are per instance, not global."""
        return None

    @classmethod
    def gen_char(cls, used: set[str] | None = None) -> str:
        """Allocate a fresh private-use character."""
        return _gen_char(set() if used is None else used)

    # -- construction / restoration --------------------------------------

    @classmethod
    def from_iterable(
        cls,
        iterable,
        text_types=None,
        mapping: dict[str, object] | None = None,
    ) -> "InlineStr":
        """Build an InlineStr from text and non-text objects.

        Strings (and instances of ``text_types``) become text; every other
        object becomes one inline segment.  InlineStr items are merged rather
        than wrapped as a single segment.
        """
        # Compatibility: Aha's old signature was (iterable, mapping=None).
        if mapping is None and isinstance(text_types, Mapping):
            mapping = text_types
            text_types = None
        text_types = (
            cls._text_types
            if text_types is None
            else _normalize_text_types(text_types)
        )
        items = list(iterable)

        used: set[str] = set()
        for item in items:
            if isinstance(item, InlineStr):
                used.update(_plain(item))
                used.update(item.mapping)
            else:
                text = _text_of(item, text_types)
                if text is not None:
                    used.update(text)

        pieces: list[str | InlineStr] = []
        for item in items:
            if isinstance(item, InlineStr):
                pieces.append(item)
                continue
            text = _text_of(item, text_types)
            if text is not None:
                pieces.append(text)
                continue
            ch = _gen_char(used)
            pieces.append(InlineStr(ch, {ch: item}, {0: item}))

        new_pieces, merged = _prepare(
            pieces, recursive=False, text_types=text_types
        )

        text_parts: list[str] = []
        indices: dict[int, object] = {}
        offset = 0
        for piece in new_pieces:
            text = _plain(piece)
            text_parts.append(text)
            if isinstance(piece, InlineStr):
                for i, ch in enumerate(text):
                    seg = piece.mapping.get(ch)
                    if seg is not None or ch in piece.mapping:
                        indices[offset + i] = seg
            offset += len(text)

        result = cls("".join(text_parts), merged, indices)
        if mapping is not None:
            # Use the final mapping: construction may have re-keyed it.
            mapping.update(result.mapping)
        return result

    @classmethod
    def from_obj(
        cls,
        obj,
        text_types=None,
        mapping: dict[str, object] | None = None,
    ) -> "InlineStr":
        """Build an InlineStr from a single object."""
        # Compatibility: Aha's old signature was (obj, mapping=None).
        if mapping is None and isinstance(text_types, Mapping):
            mapping = text_types
            text_types = None
        if isinstance(obj, InlineStr):
            if mapping is not None:
                mapping.update(obj.mapping)
            return obj
        text_types = (
            cls._text_types
            if text_types is None
            else _normalize_text_types(text_types)
        )
        text = _text_of(obj, text_types)
        if text is not None:
            return cls(text)
        ch = _gen_char(set())
        result = cls(ch, {ch: obj}, {0: obj})
        if mapping is not None:
            # Use the final mapping: construction may have re-keyed it.
            mapping.update(result.mapping)
        return result

    def to_list(self, mapping: Mapping[str, object] | None = None) -> list[object]:
        """Restore the list of text chunks and segments."""
        mp = dict(mapping) if mapping is not None else self.mapping
        text = _plain(self)
        if not mp:
            return [text]

        expected = [i for i, ch in enumerate(text) if ch in mp]
        positions: list[int]
        indices = self.inline_indices
        if indices:
            try:
                valid = all(
                    0 <= i < len(text) and text[i] in mp for i in indices
                )
            except TypeError:
                valid = False
            # Use the index map only when it covers every mapped character.
            # A partial/stale map would silently drop segments.
            positions = (
                sorted(indices)
                if valid and set(indices) == set(expected)
                else expected
            )
        else:
            positions = expected

        out: list[object] = []
        start = 0
        for i in positions:
            if i > start:
                out.append(text[start:i])
            out.append(mp[text[i]])
            start = i + 1
        if start < len(text):
            out.append(text[start:])
        return out

    # -- representation / copying ----------------------------------------

    def __str__(self) -> str:
        # Returning self (a str subclass) keeps the mapping through str(s).
        return self

    def __repr__(self) -> str:
        body = str.__repr__(self)
        if self.mapping:
            return f"{type(self).__name__}({body}, mapping={self.mapping!r})"
        return f"{type(self).__name__}({body})"

    def __reduce_ex__(self, protocol):
        return (
            type(self),
            (_plain(self), self.mapping, self.inline_indices),
        )

    def __copy__(self):
        return type(self)(_plain(self), self.mapping, self.inline_indices)

    def __deepcopy__(self, memo):
        return type(self)(
            _plain(self),
            _copy.deepcopy(self.mapping, memo),
            _copy.deepcopy(self.inline_indices, memo),
        )

    # -- operators --------------------------------------------------------

    def __add__(self, other):
        if not isinstance(other, str):
            return NotImplemented
        vals, merged = _prepare(
            [self, other], recursive=False, text_types=type(self)._text_types
        )
        a, b = vals
        text = _plain(a) + _plain(b)
        indices = {i: merged[ch] for i, ch in enumerate(text) if ch in merged}
        return type(self)(text, merged, indices)

    def __radd__(self, other):
        if not isinstance(other, str):
            return NotImplemented
        vals, merged = _prepare(
            [other, self], recursive=False, text_types=type(self)._text_types
        )
        a, b = vals
        text = _plain(a) + _plain(b)
        indices = {i: merged[ch] for i, ch in enumerate(text) if ch in merged}
        return type(self)(text, merged, indices)

    def __mul__(self, count):
        if not isinstance(count, int):
            return NotImplemented
        if count <= 0:
            return type(self)("")
        text = _plain(self) * count
        indices = {
            i: self.mapping[ch]
            for i, ch in enumerate(text)
            if ch in self.mapping
        }
        return type(self)(text, self.mapping, indices)

    __rmul__ = __mul__

    def __mod__(self, value):
        vals, merged = _prepare(
            [self, value], recursive=True, text_types=type(self)._text_types
        )
        new_self, new_value = vals
        text = str.__mod__(_plain(new_self), new_value)
        return _wrap_str(text, merged, cls=type(self))

    def __rmod__(self, value):
        vals, merged = _prepare(
            [value, self], recursive=True, text_types=type(self)._text_types
        )
        new_value, new_self = vals
        text = str.__mod__(_plain(new_value), new_self)
        return _wrap_str(text, merged, cls=type(self))

    def __format__(self, format_spec: str) -> str:
        text = str.__format__(_plain(self), format_spec)
        return _wrap_str(text, self.mapping, cls=type(self))

    def __getitem__(self, key):
        result = str.__getitem__(self, key)
        if isinstance(result, str):
            return _wrap_str(result, self.mapping, cls=type(self))
        return result

    def __iter__(self):
        for ch in str.__iter__(self):
            if ch in self.mapping:
                seg = self.mapping[ch]
                yield type(self)(ch, {ch: seg}, {0: seg})
            else:
                yield type(self)(ch)

    # -- methods with special argument handling ---------------------------

    def _prepare_call(self, args, kwargs):
        vals, merged = _prepare(
            [self, *args, *kwargs.values()],
            recursive=True,
            text_types=type(self)._text_types,
        )
        n = len(args)
        return vals[0], tuple(vals[1 : 1 + n]), dict(zip(kwargs, vals[1 + n :])), merged

    def join(self, iterable):
        items = list(iterable)
        vals, merged = _prepare(
            [self, *items], recursive=False, text_types=type(self)._text_types
        )
        new_self = vals[0]
        new_items = vals[1:]
        text = _plain(new_self).join(new_items)
        return _wrap_str(text, merged, cls=type(self))

    def split(self, sep=None, maxsplit=-1):
        if sep is None:
            vals, merged = _prepare(
                [self], recursive=False, text_types=type(self)._text_types
            )
            new_sep = None
        elif isinstance(sep, str):
            sep_text = _plain(sep)
            vals, merged = _prepare(
                [self, sep_text],
                recursive=False,
                text_types=type(self)._text_types,
            )
            new_sep = sep_text
        else:
            return _plain(self).split(sep, maxsplit)
        parts = _plain(vals[0]).split(new_sep, maxsplit)
        return [_wrap_str(p, merged, cls=type(self)) for p in parts]

    def rsplit(self, sep=None, maxsplit=-1):
        if sep is None:
            vals, merged = _prepare(
                [self], recursive=False, text_types=type(self)._text_types
            )
            new_sep = None
        elif isinstance(sep, str):
            sep_text = _plain(sep)
            vals, merged = _prepare(
                [self, sep_text],
                recursive=False,
                text_types=type(self)._text_types,
            )
            new_sep = sep_text
        else:
            return _plain(self).rsplit(sep, maxsplit)
        parts = _plain(vals[0]).rsplit(new_sep, maxsplit)
        return [_wrap_str(p, merged, cls=type(self)) for p in parts]

    def splitlines(self, keepends=False):
        vals, merged = _prepare(
            [self], recursive=False, text_types=type(self)._text_types
        )
        parts = _plain(vals[0]).splitlines(keepends)
        return [_wrap_str(p, merged, cls=type(self)) for p in parts]

    def partition(self, sep):
        if not isinstance(sep, str):
            return _plain(self).partition(sep)
        sep_text = _plain(sep)
        vals, merged = _prepare(
            [self, sep_text], recursive=False, text_types=type(self)._text_types
        )
        parts = _plain(vals[0]).partition(sep_text)
        return tuple(_wrap_str(p, merged, cls=type(self)) for p in parts)

    def rpartition(self, sep):
        if not isinstance(sep, str):
            return _plain(self).rpartition(sep)
        sep_text = _plain(sep)
        vals, merged = _prepare(
            [self, sep_text], recursive=False, text_types=type(self)._text_types
        )
        parts = _plain(vals[0]).rpartition(sep_text)
        return tuple(_wrap_str(p, merged, cls=type(self)) for p in parts)

    def strip(self, chars=None):
        if chars is None:
            vals, merged = _prepare(
                [self], recursive=False, text_types=type(self)._text_types
            )
            new_chars = None
        elif isinstance(chars, str):
            chars_text = _plain(chars)
            vals, merged = _prepare(
                [self, chars_text],
                recursive=False,
                text_types=type(self)._text_types,
            )
            new_chars = chars_text
        else:
            return _plain(self).strip(chars)
        text = _plain(vals[0]).strip(new_chars)
        return _wrap_str(text, merged, cls=type(self))

    def lstrip(self, chars=None):
        if chars is None:
            vals, merged = _prepare(
                [self], recursive=False, text_types=type(self)._text_types
            )
            new_chars = None
        elif isinstance(chars, str):
            chars_text = _plain(chars)
            vals, merged = _prepare(
                [self, chars_text],
                recursive=False,
                text_types=type(self)._text_types,
            )
            new_chars = chars_text
        else:
            return _plain(self).lstrip(chars)
        text = _plain(vals[0]).lstrip(new_chars)
        return _wrap_str(text, merged, cls=type(self))

    def rstrip(self, chars=None):
        if chars is None:
            vals, merged = _prepare(
                [self], recursive=False, text_types=type(self)._text_types
            )
            new_chars = None
        elif isinstance(chars, str):
            chars_text = _plain(chars)
            vals, merged = _prepare(
                [self, chars_text],
                recursive=False,
                text_types=type(self)._text_types,
            )
            new_chars = chars_text
        else:
            return _plain(self).rstrip(chars)
        text = _plain(vals[0]).rstrip(new_chars)
        return _wrap_str(text, merged, cls=type(self))

    def replace(self, old, new, count=-1):
        if not isinstance(old, str) or not isinstance(new, str):
            return _plain(self).replace(old, new, count)
        vals, merged = _prepare(
            [self, old, new], recursive=True, text_types=type(self)._text_types
        )
        new_self, new_old, new_new = vals
        text = _plain(new_self).replace(_plain(new_old), _plain(new_new), count)
        return _wrap_str(text, merged, cls=type(self))

    def format(self, *args, **kwargs):
        vals, merged = _prepare(
            [self, *args, *kwargs.values()],
            recursive=True,
            text_types=type(self)._text_types,
        )
        n = len(args)
        new_self = vals[0]
        new_args = tuple(vals[1 : 1 + n])
        new_kwargs = dict(zip(kwargs, vals[1 + n :]))
        text = _plain(new_self).format(*new_args, **new_kwargs)
        return _wrap_str(text, merged, cls=type(self))

    def format_map(self, mapping):
        vals, merged = _prepare(
            [self, mapping], recursive=True, text_types=type(self)._text_types
        )
        new_self, new_mapping = vals
        text = _plain(new_self).format_map(new_mapping)
        return _wrap_str(text, merged, cls=type(self))

    def translate(self, table):
        vals, merged = _prepare(
            [self, table], recursive=True, text_types=type(self)._text_types
        )
        new_self, new_table = vals
        text = _plain(new_self).translate(new_table)
        return _wrap_str(text, merged, cls=type(self))


def _make_simple_method(name: str):
    """Generate a str method that preserves mappings and their arguments."""

    def method(self, *args, **kwargs):
        vals, merged = _prepare(
            [self, *args, *kwargs.values()],
            recursive=True,
            text_types=type(self)._text_types,
        )
        n = len(args)
        new_self = vals[0]
        new_args = vals[1 : 1 + n]
        new_kwargs = dict(zip(kwargs, vals[1 + n :]))
        text = getattr(str, name)(_plain(new_self), *new_args, **new_kwargs)
        return _wrap_str(text, merged, cls=type(self))

    method.__name__ = name
    method.__qualname__ = f"InlineStr.{name}"
    method.__doc__ = f"``str.{name}`` with InlineStr propagation."
    return method


_SIMPLE_METHODS = (
    "capitalize",
    "casefold",
    "center",
    "expandtabs",
    "ljust",
    "lower",
    "removeprefix",
    "removesuffix",
    "rjust",
    "swapcase",
    "title",
    "upper",
    "zfill",
)

for _name in _SIMPLE_METHODS:
    setattr(InlineStr, _name, _make_simple_method(_name))


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


# ---------------------------------------------------------------------------
# re.Match adapter
# ---------------------------------------------------------------------------
#
# ``re.Match`` is an immutable C type: its methods cannot be monkeypatched,
# so ``m.group(n)`` always returns a fresh plain ``str``.  Inside a function
# instrumented by the AST hook (or decorated with ``inline_aware``) the
# mapping is still recovered at the return/yield boundary as long as the
# source InlineStr is in scope and the key character survived.
#
# When precise provenance is needed *inside* the function, a match group is
# always a contiguous slice of the source, so slicing the original
# InlineStr restores the mapping exactly.


def _source_slice(source, start: int, end: int):
    """Slice *source*, keeping InlineStr provenance when available."""
    if start < 0:
        return None
    return source[start:end]


class InlineMatch:
    """A ``re.Match`` proxy whose ``group``/``groups`` results are slices.

    All other attributes and methods (``start``, ``end``, ``span``, ``re``,
    ``string``, ``expand``, ...) are delegated to the real match, so this is
    a drop-in for read-only uses.
    """

    __slots__ = ("_match", "_source")

    def __init__(self, match, source=None):
        # ``re.Match.string`` is the exact object passed to ``re``, so it is
        # already the InlineStr when one was used.
        self._match = match
        self._source = match.string if source is None else source

    def group(self, *groups):
        if not groups:
            groups = (0,)
        out = [self._group(g) for g in groups]
        return out[0] if len(out) == 1 else tuple(out)

    def groups(self, default=None):
        return tuple(self._group(i, default) for i in range(1, self._match.re.groups + 1))

    def groupdict(self, default=None):
        return {
            name: self._group(name, default)
            for name in self._match.re.groupindex
        }

    def _group(self, group, default=None):
        span = self._match.span(group)
        if span == (-1, -1):
            return default
        return _source_slice(self._source, span[0], span[1])

    def __getattr__(self, name):
        return getattr(self._match, name)

    def __repr__(self):
        return f"InlineMatch({self._match!r})"


def inline_match(match, source=None) -> InlineMatch:
    """Wrap a ``re.Match`` so group results are slices of *source*.

    *source* defaults to ``match.string``, which is the exact object passed
    to ``re`` (an InlineStr when one was used).
    """
    return InlineMatch(match, source)


def inline_group(match, source, group=0):
    """Return match group *group* as a slice of *source* (no proxy)."""
    start, end = match.span(group)
    return _source_slice(source, start, end)


def inline_groups(match, source, default=None) -> tuple:
    """Return all groups as slices of *source* (no proxy)."""
    return tuple(
        _source_slice(source, *match.span(i)) if match.span(i) != (-1, -1) else default
        for i in range(1, match.re.groups + 1)
    )


def inline_finditer(pattern, string, flags=0):
    """``re.finditer`` yielding :class:`InlineMatch` proxies."""
    for match in re.finditer(pattern, string, flags):
        yield InlineMatch(match, string)


def inline_findall(pattern, string, flags=0) -> list:
    """``re.findall`` with results sliced from *string*."""
    out: list = []
    for match in re.finditer(pattern, string, flags):
        groups = match.re.groups
        if groups == 0:
            out.append(_source_slice(string, *match.span(0)))
        elif groups == 1:
            out.append(_source_slice(string, *match.span(1)))
        else:
            out.append(
                tuple(_source_slice(string, *match.span(i)) for i in range(1, groups + 1))
            )
    return out


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


def _match_group_adapter(obj, name, args, kwargs):
    """Default adapter: recover ``re.Match`` groups over InlineStr sources."""
    if isinstance(obj, re.Match) and isinstance(obj.string, InlineStr):
        return getattr(InlineMatch(obj), name)(*args, **kwargs)
    return NOT_HANDLED


for _match_method_name in DEFAULT_MATCH_METHODS:
    register_method_adapter(_match_method_name, _match_group_adapter)
del _match_method_name


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


# ---------------------------------------------------------------------------
# AST import hook
# ---------------------------------------------------------------------------

_RUNTIME_NAME = "_inline_str_rt__"
_PREPARED_NAME = "_inline_str_prepared__"
_VARLEN_NAME = "_inline_str_varlen__"


def _name(name: str, ctx: ast.expr_context | None = None) -> ast.Name:
    return ast.Name(id=name, ctx=ctx or ast.Load())


def _const(value) -> ast.Constant:
    return ast.Constant(value=value)


def _attr(value: ast.expr, attr: str) -> ast.Attribute:
    return ast.Attribute(value=value, attr=attr, ctx=ast.Load())


def _call(func: ast.expr, args: list[ast.expr]) -> ast.Call:
    return ast.Call(func=func, args=args, keywords=[])


class _ReturnWrapper(ast.NodeTransformer):
    """Wrap return/yield values with the runtime recovery helpers."""

    def visit_FunctionDef(self, node: ast.FunctionDef):
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return node

    def visit_Lambda(self, node: ast.Lambda):
        return node

    def visit_Return(self, node: ast.Return):
        if node.value is None:
            return node
        wrapped = _call(
            _attr(_name(_RUNTIME_NAME), "wrap_return_here"), [node.value]
        )
        node.value = ast.copy_location(wrapped, node.value)
        return node

    def visit_Yield(self, node: ast.Yield):
        if node.value is None:
            return node
        wrapped = _call(
            _attr(_name(_RUNTIME_NAME), "wrap_return_here"), [node.value]
        )
        node.value = ast.copy_location(wrapped, node.value)
        return node

    def visit_YieldFrom(self, node: ast.YieldFrom):
        # ``yield from`` must wrap each yielded item and preserve send/throw.
        wrapped = _call(
            _attr(_name(_RUNTIME_NAME), "wrap_iterable"),
            [
                node.value,
                _call(_attr(_name(_RUNTIME_NAME), "scope_here"), []),
            ],
        )
        node.value = ast.copy_location(wrapped, node.value)
        return node


class InlineTransformer(ast.NodeTransformer):
    def __init__(
        self,
        runtime: str,
        rewrite_methods=DEFAULT_MATCH_METHODS,
        rewrite_calls: bool = False,
    ):
        self.runtime = runtime
        if rewrite_methods is not True and rewrite_methods is not False:
            rewrite_methods = frozenset(rewrite_methods or ())
        self.rewrite_methods = rewrite_methods
        self.rewrite_calls = rewrite_calls

    def _method_selected(self, name: str) -> bool:
        selected = self.rewrite_methods
        if selected is True:
            return True
        if not selected:
            return False
        return name in selected

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        func = node.func
        if isinstance(func, ast.Attribute) and self._method_selected(func.attr):
            # ``obj.name(x)`` -> ``_rt._dispatch_method(obj, "name", x)``.
            # The dispatcher only replaces the result when an adapter
            # registered for that name returns something other than
            # NOT_HANDLED; otherwise the real method runs unchanged.
            call = ast.Call(
                func=_attr(_name(_RUNTIME_NAME), "_dispatch_method"),
                args=[
                    func.value,
                    ast.Constant(func.attr),
                    *node.args,
                ],
                keywords=node.keywords,
            )
            return ast.copy_location(call, node)
        if self.rewrite_calls:
            # ``f(x)`` -> ``_rt._dispatch_call(f, x)`` for every remaining
            # call.  This is opt-in: it adds one Python-level dispatch per
            # call in the instrumented module.
            call = ast.Call(
                func=_attr(_name(_RUNTIME_NAME), "_dispatch_call"),
                args=[func, *node.args],
                keywords=node.keywords,
            )
            return ast.copy_location(call, node)
        return node

    def visit_Module(self, node: ast.Module):
        self.generic_visit(node)
        imp = ast.Import(
            names=[ast.alias(name=self.runtime, asname=_RUNTIME_NAME)]
        )
        index = 1 if _is_docstring(node.body[0] if node.body else None) else 0
        while (
            index < len(node.body)
            and isinstance(node.body[index], ast.ImportFrom)
            and node.body[index].module == "__future__"
        ):
            index += 1
        node.body.insert(index, imp)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.generic_visit(node)
        return self._transform(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda):
        # A lambda has no statement body to insert a prologue into, so wrap it
        # with the same decorator used for explicit boundaries.  The wrapper
        # also recovers mappings from closure cells.
        self.generic_visit(node)
        wrapped = _call(
            _attr(_name(_RUNTIME_NAME), "inline_aware"), [node]
        )
        return ast.copy_location(wrapped, node)

    def _transform(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        # 1. Every return gets a recovery wrapper.
        for i, stmt in enumerate(node.body):
            node.body[i] = _ReturnWrapper().visit(stmt)

        args = node.args
        normal = [
            a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)
        ]
        vararg = args.vararg.arg if args.vararg else None
        kwarg = args.kwarg.arg if args.kwarg else None
        if not (normal or vararg or kwarg):
            return node

        prologue: list[ast.stmt] = []

        # 2. Entry: prepare_many((p1, ..., *args, *kwargs.values()))
        if vararg:
            prologue.append(
                ast.Assign(
                    targets=[_name(_VARLEN_NAME, ast.Store())],
                    value=_call(_name("len"), [_name(vararg)]),
                )
            )
        elements: list[ast.expr] = [_name(p) for p in normal]
        if vararg:
            elements.append(ast.Starred(value=_name(vararg), ctx=ast.Load()))
        if kwarg:
            elements.append(
                ast.Starred(
                    value=_call(_attr(_name(kwarg), "values"), []),
                    ctx=ast.Load(),
                )
            )
        prologue.append(
            ast.Assign(
                targets=[_name(_PREPARED_NAME, ast.Store())],
                value=ast.Call(
                    func=_attr(_name(_RUNTIME_NAME), "prepare_many"),
                    args=[ast.Tuple(elts=elements, ctx=ast.Load())],
                    keywords=[
                        ast.keyword(
                            arg="with_scope", value=ast.Constant(True)
                        )
                    ],
                ),
            )
        )

        # 3. Write the possibly re-keyed values back into the parameters.
        assignments: list[ast.stmt] = []
        n = len(normal)
        if normal:
            target = ast.Tuple(
                elts=[_name(p, ast.Store()) for p in normal],
                ctx=ast.Store(),
            )
            if vararg or kwarg:
                value: ast.expr = ast.Subscript(
                    value=_name(_PREPARED_NAME),
                    slice=ast.Slice(lower=None, upper=_const(n), step=None),
                    ctx=ast.Load(),
                )
            else:
                value = _name(_PREPARED_NAME)
            assignments.append(ast.Assign(targets=[target], value=value))
        if vararg:
            lower = _const(n)
            upper: ast.expr = (
                ast.BinOp(
                    left=_name(_VARLEN_NAME), op=ast.Add(), right=_const(n)
                )
                if n
                else _name(_VARLEN_NAME)
            )
            assignments.append(
                ast.Assign(
                    targets=[_name(vararg, ast.Store())],
                    value=ast.Subscript(
                        value=_name(_PREPARED_NAME),
                        slice=ast.Slice(lower=lower, upper=upper, step=None),
                        ctx=ast.Load(),
                    ),
                )
            )
        if kwarg:
            start: ast.expr = (
                ast.BinOp(
                    left=_const(n),
                    op=ast.Add(),
                    right=_name(_VARLEN_NAME),
                )
                if vararg
                else _const(n)
            )
            assignments.append(
                ast.Assign(
                    targets=[_name(kwarg, ast.Store())],
                    value=_call(
                        _name("dict"),
                        [
                            _call(
                                _name("zip"),
                                [
                                    _name(kwarg),
                                    ast.Subscript(
                                        value=_name(_PREPARED_NAME),
                                        slice=ast.Slice(
                                            lower=start, upper=None, step=None
                                        ),
                                        ctx=ast.Load(),
                                    ),
                                ],
                            )
                        ],
                    ),
                )
            )
        if assignments:
            prologue.append(
                ast.If(
                    test=ast.Compare(
                        left=_name(_PREPARED_NAME),
                        ops=[ast.IsNot()],
                        comparators=[_const(None)],
                    ),
                    body=assignments,
                    orelse=[],
                )
            )

        index = 1 if _is_docstring(node.body[0] if node.body else None) else 0
        location = node.body[index] if index < len(node.body) else node.body[-1]
        for stmt in prologue:
            ast.copy_location(stmt, location)
        node.body[index:index] = prologue
        return node


def _is_docstring(stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


class _Loader(importlib.abc.SourceLoader):
    def __init__(
        self,
        fullname: str,
        path: str,
        runtime: str,
        rewrite_methods=DEFAULT_MATCH_METHODS,
        rewrite_calls: bool = False,
    ):
        self.fullname = fullname
        self.path = path
        self.runtime = runtime
        self.rewrite_methods = rewrite_methods
        self.rewrite_calls = rewrite_calls

    def get_filename(self, fullname: str) -> str:
        return self.path

    def get_data(self, path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def get_code(self, fullname: str):
        # Bypass .pyc: a cached bytecode file would skip the transformer.
        return self.source_to_code(self.get_data(self.path), self.path)

    def source_to_code(self, data, path, *, _optimize=-1):
        source = importlib.util.decode_source(data)
        tree = ast.parse(source, path)
        tree = InlineTransformer(
            self.runtime,
            rewrite_methods=self.rewrite_methods,
            rewrite_calls=self.rewrite_calls,
        ).visit(tree)
        ast.fix_missing_locations(tree)
        return compile(tree, path, "exec", dont_inherit=True)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(
        self,
        runtime: str,
        prefixes: tuple[str, ...],
        rewrite_methods=DEFAULT_MATCH_METHODS,
        rewrite_calls: bool = False,
    ):
        self.runtime = runtime
        self.prefixes = prefixes
        self.rewrite_methods = rewrite_methods
        self.rewrite_calls = rewrite_calls

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.runtime:
            return None
        if self.prefixes and "*" not in self.prefixes:
            if not any(
                fullname == p or fullname.startswith(p + ".")
                for p in self.prefixes
            ):
                return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            return spec
        spec.loader = _Loader(
            fullname,
            spec.origin,
            self.runtime,
            rewrite_methods=self.rewrite_methods,
            rewrite_calls=self.rewrite_calls,
        )
        return spec


def install_ast_hook(
    runtime: str = __name__,
    *prefixes: str,
    rewrite_methods=DEFAULT_MATCH_METHODS,
    rewrite_calls: bool = False,
) -> None:
    """Install the import hook for *prefixes*.

    ``runtime`` must be the importable module name containing this file.
    Install before importing the target modules.  Pass ``"*"`` (or no
    prefixes at all) to instrument every pure-Python module imported
    afterwards -- powerful, but it also rewrites third-party code, so use it
    only in a controlled process.

    Call-site rewriting is configured by two parameters:

    ``rewrite_methods``
        ``False`` disables method rewriting; ``True`` rewrites every
        ``obj.name(...)`` call; an iterable of names rewrites only those
        names.  Defaults to :data:`DEFAULT_MATCH_METHODS`
        (``group`` / ``groups`` / ``groupdict``).

    ``rewrite_calls``
        ``True`` additionally rewrites every plain ``f(...)`` call (including
        method calls whose name was not selected above) to
        :func:`_dispatch_call`, enabling blanket call hooking.  This costs one
        runtime dispatch per call, so it is opt-in.

    Rewritten calls consult the adapter registries populated by
    :func:`register_method_adapter` / :func:`register_call_adapter`.  The
    default match adapter is itself registered through that API, so it can be
    removed with ``unregister_method_adapter("group", _match_group_adapter)``
    or simply by disabling rewriting.

    Example: adapt a custom ``.render()`` method::

        register_method_adapter("render", my_render_adapter)
        install_ast_hook(__name__, "my_package",
                         rewrite_methods=DEFAULT_MATCH_METHODS | {"render"})
    """
    sys.meta_path.insert(
        0,
        _Finder(
            runtime,
            prefixes,
            rewrite_methods=rewrite_methods,
            rewrite_calls=rewrite_calls,
        ),
    )
    importlib.invalidate_caches()


def _mark_aware(func):
    try:
        func.__inline_aware__ = True
    except Exception:
        pass
    return func


def patch_module(module, names=None, include_classes: bool = True):
    """Patch functions/methods of an already-imported module in place.

    This is the fallback for modules that were imported before the AST hook
    was installed, or that cannot be AST-transformed.  It wraps at the Python
    call boundary: arguments are prepared and plain-str results are recovered
    from the arguments/closures.  It cannot rewrite the function's internals,
    so a mapping created only from a local (not from an argument/closure)
    still needs the AST hook or an explicit ``@inline_aware``.

    ``module`` may be a module object or an importable name.  ``names`` limits
    the patch to specific attributes; by default public functions and public
    methods defined by the module/class are patched.  Returns *module*.
    """
    if isinstance(module, str):
        module = importlib.import_module(module)
    modname = getattr(module, "__name__", None)
    explicit = names is not None
    if names is None:
        names = [n for n in dir(module) if not n.startswith("_")]
    elif isinstance(names, str):
        names = [names]

    def should_patch(obj):
        if not callable(obj):
            return False
        return explicit or getattr(obj, "__module__", None) == modname

    for name in names:
        try:
            obj = getattr(module, name)
        except AttributeError:
            continue
        if inspect.isfunction(obj) or inspect.isbuiltin(obj):
            if should_patch(obj) and not getattr(obj, "__inline_aware__", False):
                setattr(module, name, _mark_aware(inline_aware(obj)))
        elif include_classes and inspect.isclass(obj):
            if not explicit and getattr(obj, "__module__", None) != modname:
                continue
            for attr_name, attr in list(vars(obj).items()):
                if not explicit and attr_name.startswith("__") and attr_name.endswith("__"):
                    continue
                if inspect.isfunction(attr):
                    if not getattr(attr, "__inline_aware__", False):
                        setattr(obj, attr_name, _mark_aware(inline_aware(attr)))
                elif isinstance(attr, staticmethod):
                    func = attr.__func__
                    if inspect.isfunction(func) and not getattr(func, "__inline_aware__", False):
                        setattr(obj, attr_name, staticmethod(_mark_aware(inline_aware(func))))
                elif isinstance(attr, classmethod):
                    func = attr.__func__
                    if inspect.isfunction(func) and not getattr(func, "__inline_aware__", False):
                        setattr(obj, attr_name, classmethod(_mark_aware(inline_aware(func))))
                elif isinstance(attr, property) and attr.fget is not None:
                    fget = attr.fget
                    if not getattr(fget, "__inline_aware__", False):
                        setattr(
                            obj,
                            attr_name,
                            property(
                                _mark_aware(inline_aware(fget)),
                                attr.fset,
                                attr.fdel,
                                attr.__doc__,
                            ),
                        )
    return module


# ---------------------------------------------------------------------------
# smoke test / demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    class Seg:
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return f"Seg({self.name!r})"

    seg = Seg("at")
    s = InlineStr.from_iterable(["hello ", seg, " world"])
    assert s.to_list() == ["hello ", seg, " world"]
    assert (s + "!").to_list() == ["hello ", seg, " world!"]
    assert s.upper().to_list() == ["HELLO ", seg, " WORLD"]
    assert s.split(" ")[0].to_list() == ["hello"]
    assert s.replace("world", "there").to_list() == [
        "hello ",
        seg,
        " there",
    ]
    assert "{}".format(s).to_list() == ["hello ", seg, " world"]
    assert f"{s}".to_list() == ["hello ", seg, " world"]
    assert ("pre%spost" % s).to_list() == [
        "prehello ",
        seg,
        " worldpost",
    ]
    print("self-test ok:", s.to_list())
