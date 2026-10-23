"""AST import hook for instrumenting Python modules at import time."""

from __future__ import annotations

import ast
import importlib.abc
import importlib.machinery
import sys

from ._adapters import DEFAULT_MATCH_METHODS


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
        # Never instrument the runtime package itself (including submodules
        # such as ``aware_inline_str.re``), even when "*" is requested.
        if fullname == self.runtime or fullname.startswith(self.runtime + "."):
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
    default match adapter lives in :mod:`aware_inline_str.re` and is itself
    registered through that API, so ``unregister_method_adapter("group")``
    removes it, or simply disable rewriting.

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
