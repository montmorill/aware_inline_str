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
  Uninstrumented code can additionally use the helpers in
  :mod:`aware_inline_str.re` (``inline_match(m)``, ``inline_group``,
  ``inline_groups``, ``inline_finditer``, ``inline_findall``) -- a group is a
  contiguous slice of the source, so slicing the original InlineStr restores
  its mapping exactly.
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
  key rather than misattributing it; the adapter helpers in
  ``aware_inline_str.re`` (``inline_group`` etc.) and explicitly prepared call
  sites avoid the situation.

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

from ._adapters import (
    DEFAULT_MATCH_METHODS,
    NOT_HANDLED,
    _dispatch_call,
    _dispatch_method,
    register_call_adapter,
    register_method_adapter,
    unregister_call_adapter,
    unregister_method_adapter,
)
from ._ast import install_ast_hook
from ._core import CodePointConflictError, InlineStr
from ._decorator import inline_aware
from ._patch import patch_module
from ._recover import (
    prepare_many,
    scope_here,
    wrap_iterable,
    wrap_return,
    wrap_return_here,
)

__all__ = [
    "InlineStr",
    "CodePointConflictError",
    "DEFAULT_MATCH_METHODS",
    "NOT_HANDLED",
    "register_call_adapter",
    "register_method_adapter",
    "unregister_call_adapter",
    "unregister_method_adapter",
    "inline_aware",
    "install_ast_hook",
    "patch_module",
    "prepare_many",
    "scope_here",
    "wrap_iterable",
    "wrap_return",
    "wrap_return_here",
]

# Import the re helpers last: this registers the default ``group`` /
# ``groups`` / ``groupdict`` adapters and exposes ``aware_inline_str.re``.
# The name ``re`` is the submodule, not the stdlib module.
from . import re as re  # noqa: E402,F401
