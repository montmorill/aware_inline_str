"""``re.Match`` helpers and the default match adapters.

``re.Match`` is an immutable C type: its methods cannot be monkeypatched, so
``m.group(n)`` always returns a fresh plain ``str``.  Inside a function
instrumented by the AST hook (or decorated with ``inline_aware``) the mapping
is still recovered at the return/yield boundary as long as the source
InlineStr is in scope and the key character survived.

When precise provenance is needed *inside* the function, a match group is
always a contiguous slice of the source, so slicing the original InlineStr
restores the mapping exactly.  That is what the helpers here do.

Importing this module also registers the default ``group`` / ``groups`` /
``groupdict`` method adapters, so ``install_ast_hook`` can rewrite match
calls in instrumented code.  The package imports this module automatically;
``import aware_inline_str.re`` is only needed to use the explicit helpers.
"""

from __future__ import annotations

import re

from ._adapters import (
    DEFAULT_MATCH_METHODS,
    NOT_HANDLED,
    register_method_adapter,
)
from ._core import InlineStr

__all__ = [
    "InlineMatch",
    "inline_findall",
    "inline_finditer",
    "inline_group",
    "inline_groups",
    "inline_match",
]


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


def _match_group_adapter(obj, name, args, kwargs):
    """Default adapter: recover ``re.Match`` groups over InlineStr sources."""
    if isinstance(obj, re.Match) and isinstance(obj.string, InlineStr):
        return getattr(InlineMatch(obj), name)(*args, **kwargs)
    return NOT_HANDLED


for _match_method_name in DEFAULT_MATCH_METHODS:
    register_method_adapter(_match_method_name, _match_group_adapter)
del _match_method_name
