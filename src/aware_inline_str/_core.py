"""Core implementation: code points, conflicts and the InlineStr class."""

from __future__ import annotations

import copy as _copy
import dataclasses
import itertools
import random
import threading
import types
import weakref
from collections.abc import Mapping


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
