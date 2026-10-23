"""Call-boundary patching for already-imported modules."""

from __future__ import annotations

import importlib
import inspect

from ._decorator import inline_aware


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
