"""End-to-end tests for the AST hook, @inline_aware and call adapters.

The instrumented fixture packages are generated in a temporary directory, so
this file is self-contained and can be run from anywhere next to the
``src`` directory:

    python test_ast.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile

# Run from a source checkout without installing (src layout).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from aware_inline_str import (
    DEFAULT_MATCH_METHODS,
    NOT_HANDLED,
    InlineStr,
    inline_aware,
    install_ast_hook,
    patch_module,
    prepare_many,
    register_call_adapter,
    register_method_adapter,
    wrap_return,
)
from aware_inline_str.re import (
    inline_findall,
    inline_group,
    inline_groups,
    inline_match,
)

# ---------------------------------------------------------------------------
# fixture sources
# ---------------------------------------------------------------------------

PKG_SOURCE = r'''
import json
import re
from collections import namedtuple
from dataclasses import dataclass
from types import SimpleNamespace


def plain_join(a, b):
    return "".join([a, "|", b])


def regex_sub(a):
    return re.sub("A", "X", a)


def fstring_multi(a, b):
    return f"pre{a}{b}post"


def container(a, b):
    return [a, "|", b]


def nested_container(a):
    return {"x": [a, ("y", a)]}


def attr_value(a):
    ns = SimpleNamespace(s=a)
    return ns.s


def dataclass_return(a):
    @dataclass
    class Box:
        s: object
        n: int = 1
    return Box(a)


def namedtuple_return(a):
    Pair = namedtuple("Pair", "s n")
    return Pair(a, 1)


def lambda_outer(a):
    return (lambda x: "".join([x, "?"]))(a)


def lambda_closure(a):
    f = lambda: "".join([a, "!"])
    return f()


def lambda_container(a):
    return (lambda x: [x, "tail"])(a)


def generator(a):
    yield "start:"
    yield "".join([a, "!"])


def generator_yield_from(a):
    yield from ["begin:", "".join([a, "!"])]


def generator_send(a):
    x = yield "ready"
    yield "".join([x, a])


async def async_gen(a):
    yield "".join([a, "!"])


def json_keep(a):
    return json.dumps({"x": a}, ensure_ascii=False)


def json_escape(a):
    return json.dumps({"x": a}, ensure_ascii=True)


def lambda_conflict(a, b):
    return (lambda x, y: f"{x}|{y}")(a, b)


def lambda_default(a):
    f = lambda x=a: "".join([x, "!"])
    return f()


def lambda_kw(a):
    f = lambda *, x: "".join([x, "!"])
    return f(x=a)


def lambda_nested(a):
    return (lambda: (lambda x: "".join([x, "?"]))(a))()


def lambda_gen(a):
    f = lambda: (yield "".join([a, "!"]))
    return f()


def inner_yield_from(a):
    yield "inner-start"
    return "".join([a, "!"])


def outer_yield_from(a):
    result = yield from inner_yield_from(a)
    return result


def generator_throw(a):
    try:
        yield "ready"
    except ValueError:
        yield "".join([a, "!"])


def generator_close(a):
    yield "".join([a, "!"])


class Plain:
    def __init__(self, s):
        self.s = s

    def join_attr(self):
        return "".join([self.s, "!"])

    @property
    def prop(self):
        return "".join([self.s, "?"])


def regex_group(a):
    m = re.match(r"(.)(.*)", a)
    return m.group(2)


def regex_group_chain(a):
    m = re.match(r"(.+?)(B)", a)
    return m.group(1).upper() + m.group(2).lower()


class Other:
    def __repr__(self):
        return "Other()"


OTHER = Other()


def internal_conflict(a):
    """Build a conflicting InlineStr *inside* the function."""
    ch = next(iter(a.mapping))
    b = type(a)("C" + ch + "D", {ch: OTHER})
    return "".join([a, b])


def group_inside(a):
    m = re.search(r"(A)(.B)", a)
    g = m.group(2)
    return [type(g).__name__, g.to_list()]


def groups_inside(a):
    m = re.search(r"(?P<head>A)(?P<rest>.B)", a)
    gd = m.groupdict()
    gs = m.groups()
    return [gd["rest"].to_list(), gs[1].to_list()]


class GroupHolder:
    def group(self, value):
        return f"plain:{value}"


def unrelated_group():
    return GroupHolder().group(3)


class Slicer:
    """A custom API adapted through a user-registered method adapter."""

    def __init__(self, value):
        self.value = value

    def take(self, n):
        return str(self.value)[:n]


def use_take(a):
    return Slicer(a).take(2)


def use_take_plain():
    return Slicer("plain").take(2)
'''

PLAIN_SOURCE = r'''
import json
import re
from dataclasses import dataclass


def combine(a, b):
    return "".join([a, "|", b])


def regex_sub(a):
    return re.sub("A", "X", a)


def json_keep(a):
    return json.dumps({"x": a}, ensure_ascii=False)


def helper_local(a):
    x = a.upper()
    return x


@dataclass
class Box:
    s: object

    def render(self):
        return "[" + self.s + "]"

    @property
    def value(self):
        return self.s

    @staticmethod
    def static(a):
        return "".join([a, "!"])


class PlainBox:
    def __init__(self, s):
        self.s = s

    def render(self):
        return "[" + self.s + "]"
'''

ALLCALLS_SOURCE = r'''
class Holder:
    def __init__(self, value):
        self.value = value

    def combine(self, suffix):
        return str(self.value) + suffix


def helper(value):
    return "helper:" + value


def run(value):
    return helper(value)


def run_method(value):
    return Holder(value).combine("!")


def run_nested(value):
    return helper(helper(value))
'''


# ---------------------------------------------------------------------------
# user-registered adapters
# ---------------------------------------------------------------------------

def take_adapter(obj, name, args, kwargs):
    """Method adapter: slice the receiver's InlineStr value."""
    value = getattr(obj, "value", None)
    if isinstance(value, InlineStr):
        return value[: args[0]]
    return NOT_HANDLED


register_method_adapter("take", take_adapter)

_seen_calls = []


def helper_adapter(func, name, args, kwargs):
    """Call adapter matched by function name."""
    if args and isinstance(args[0], InlineStr) and args[0].mapping:
        return args[0][:1] + "?"
    return NOT_HANDLED


def combine_adapter(func, name, args, kwargs):
    """Method call adapted through the all-calls path (bound method)."""
    receiver = getattr(func, "__self__", None)
    value = getattr(receiver, "value", None)
    if isinstance(value, InlineStr):
        return value[:1]
    return NOT_HANDLED


def spy_adapter(func, name, args, kwargs):
    """Wildcard adapter: observe calls without replacing them."""
    _seen_calls.append(name)
    return NOT_HANDLED


register_call_adapter("helper", helper_adapter)
register_call_adapter("combine", combine_adapter)
register_call_adapter("*", spy_adapter)


class Seg:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Seg({self.name!r})"

    def __eq__(self, other):
        return isinstance(other, Seg) and other.name == self.name

    def __hash__(self):
        return hash(self.name)


def check(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")
    print(f"ok {label}")


def _write_fixture(root, package, source):
    directory = os.path.join(root, package)
    os.makedirs(directory)
    with open(os.path.join(directory, "__init__.py"), "w", encoding="utf-8"):
        pass
    with open(os.path.join(directory, "mod.py"), "w", encoding="utf-8") as f:
        f.write(source)


def main() -> None:
    root = tempfile.mkdtemp(prefix="aware_inline_test_")
    sys.path.insert(0, root)
    try:
        _write_fixture(root, "aware_test_pkg", PKG_SOURCE)
        _write_fixture(root, "aware_test_plain", PLAIN_SOURCE)
        _write_fixture(root, "aware_test_allcalls", ALLCALLS_SOURCE)

        # ``take`` is not known to the runtime: it is added through manual
        # configuration of the call-site rewrite.
        install_ast_hook(
            "aware_inline_str",
            "aware_test_pkg",
            rewrite_methods=DEFAULT_MATCH_METHODS | {"take"},
        )
        # A second hook demonstrates blanket call hooking for another package.
        install_ast_hook(
            "aware_inline_str",
            "aware_test_allcalls",
            rewrite_methods=False,
            rewrite_calls=True,
        )

        import aware_test_allcalls.mod as m3
        import aware_test_pkg.mod as m
        import aware_test_plain.mod as u

        one = Seg("one")
        two = Seg("two")
        k = ""
        a = InlineStr(f"A{k}B", {k: one})
        b = InlineStr(f"C{k}D", {k: two})

        # -- function-boundary recovery through C-level operations --------
        check("plain_join", m.plain_join(a, b).to_list(), ["A", one, "B|C", two, "D"])
        check("regex_sub", m.regex_sub(a).to_list(), ["X", one, "B"])
        check(
            "fstring_multi",
            m.fstring_multi(a, b).to_list(),
            ["preA", one, "BC", two, "Dpost"],
        )
        check(
            "container",
            [x.to_list() for x in m.container(a, b)],
            [["A", one, "B"], ["|"], ["C", two, "D"]],
        )
        nested = m.nested_container(a)
        check("nested list", nested["x"][0].to_list(), ["A", one, "B"])
        check("nested tuple", nested["x"][1][1].to_list(), ["A", one, "B"])
        check("attr_value", m.attr_value(a).to_list(), ["A", one, "B"])
        box = m.dataclass_return(a)
        check("dataclass type", type(box).__name__, "Box")
        check("dataclass attr", box.s.to_list(), ["A", one, "B"])
        nt = m.namedtuple_return(a)
        check("namedtuple type", type(nt)._fields, ("s", "n"))
        check("namedtuple attr", nt.s.to_list(), ["A", one, "B"])

        # -- lambdas ------------------------------------------------------
        check("lambda_outer", m.lambda_outer(a).to_list(), ["A", one, "B?"])
        check("lambda_closure", m.lambda_closure(a).to_list(), ["A", one, "B!"])
        check(
            "lambda_container",
            [x.to_list() for x in m.lambda_container(a)],
            [["A", one, "B"], ["tail"]],
        )
        check(
            "lambda_conflict",
            m.lambda_conflict(a, b).to_list(),
            ["A", one, "B|C", two, "D"],
        )
        check("lambda_default", m.lambda_default(a).to_list(), ["A", one, "B!"])
        check("lambda_kw", m.lambda_kw(a).to_list(), ["A", one, "B!"])
        check("lambda_nested", m.lambda_nested(a).to_list(), ["A", one, "B?"])
        check(
            "lambda_gen",
            [x.to_list() for x in m.lambda_gen(a)],
            [["A", one, "B!"]],
        )

        # -- yield / yield from / send / throw / close --------------------
        check(
            "generator",
            [x.to_list() for x in m.generator(a)],
            [["start:"], ["A", one, "B!"]],
        )
        check(
            "generator_yield_from",
            [x.to_list() for x in m.generator_yield_from(a)],
            [["begin:"], ["A", one, "B!"]],
        )
        g = m.generator_send(a)
        check("generator_send first", next(g).to_list(), ["ready"])
        check("generator_send second", g.send("S").to_list(), ["SA", one, "B"])

        g = m.outer_yield_from(a)
        check("outer_yield_from item", next(g).to_list(), ["inner-start"])
        try:
            next(g)
        except StopIteration as exc:
            check("outer_yield_from return", exc.value.to_list(), ["A", one, "B!"])
        else:
            raise AssertionError("outer_yield_from did not stop")

        g = m.generator_throw(a)
        check("generator_throw first", next(g).to_list(), ["ready"])
        check("generator_throw second", g.throw(ValueError).to_list(), ["A", one, "B!"])

        g = m.generator_close(a)
        check("generator_close item", next(g).to_list(), ["A", one, "B!"])
        g.close()

        # -- JSON ---------------------------------------------------------
        check(
            "json_keep",
            m.json_keep(a).to_list(),
            ['{"x": "A', one, 'B"}'],
        )
        escaped = m.json_escape(a)
        check("json_escape type", isinstance(escaped, InlineStr), True)
        check("json_escape mapping", escaped.mapping, {})

        # -- async generator ----------------------------------------------
        async def run_async():
            return [x.to_list() async for x in m.async_gen(a)]

        check("async_gen", asyncio.run(run_async()), [["A", one, "B!"]])

        # -- attributes (opt-in scanning) ---------------------------------
        plain = m.Plain(a)
        check("attr before scan", isinstance(plain.join_attr(), InlineStr), False)
        InlineStr.configure(scan_objects=True)
        try:
            check("attr after scan", plain.join_attr().to_list(), ["A", one, "B!"])
            check("property after scan", plain.prop.to_list(), ["A", one, "B?"])
        finally:
            InlineStr.configure(scan_objects=False)

        # -- untransformed module: patch at the Python call boundary ------
        patch_module(u)
        check("patched combine", u.combine(a, b).to_list(), ["A", one, "B|C", two, "D"])
        check("patched regex", u.regex_sub(a).to_list(), ["X", one, "B"])
        check("patched json", u.json_keep(a).to_list(), ['{"x": "A', one, 'B"}'])
        check("patched helper", u.helper_local(a).to_list(), ["A", one, "B"])
        check("patched dataclass method", u.Box.render(u.Box(a)).to_list(), ["[A", one, "B]"])
        check("patched property", u.Box.value.__get__(u.Box(a)).to_list(), ["A", one, "B"])
        check("patched staticmethod", u.Box.static(a).to_list(), ["A", one, "B!"])
        check("patched plain method", u.PlainBox.render(u.PlainBox(a)).to_list(), ["[A", one, "B]"])

        # -- direct runtime helpers ---------------------------------------
        check("prepare_many none", prepare_many((a, "x")), None)
        InlineStr.configure(key_registry=False)
        try:
            c = InlineStr(f"C{k}D", {k: two})
        finally:
            InlineStr.configure(key_registry=True)
        prepared = prepare_many((a, c))
        if prepared is None:
            raise AssertionError("prepare_many should re-key conflicting values")
        check(
            "prepare_many conflict",
            [x.to_list() if isinstance(x, InlineStr) else x for x in prepared],
            [["A", one, "B"], ["C", two, "D"]],
        )

        # -- construction-time conflict prevention ------------------------
        live = InlineStr(f"X{k}Y", {k: one})
        clash = InlineStr(f"Z{k}W", {k: two})
        check("constructor clash segments", clash.to_list(), ["Z", two, "W"])
        check("constructor keys differ", set(live.mapping).isdisjoint(clash.mapping), True)
        same = InlineStr(f"Q{k}R", {k: one})
        check("constructor same object keeps key", k in same.mapping, True)

        # -- entry-time preparation against a closure value ---------------
        InlineStr.configure(key_registry=False)
        try:
            clo_one = InlineStr(f"A{k}B", {k: one})
            clo_two = InlineStr(f"C{k}D", {k: two})
        finally:
            InlineStr.configure(key_registry=True)

        @inline_aware
        def closure_conflict(arg):
            return "".join([arg, clo_two])

        check(
            "closure conflict",
            closure_conflict(clo_one).to_list(),
            ["A", one, "BC", two, "D"],
        )
        check(
            "wrap_return list",
            [x.to_list() for x in wrap_return(["x", a], {"a": a})],
            [["x"], ["A", one, "B"]],
        )
        check("wrap_return str", wrap_return("A" + k + "B", {"a": a}).to_list(), ["A", one, "B"])

        set_out = wrap_return({"A" + k + "B"}, {"a": a})
        check("wrap_return set", [x.to_list() for x in set_out], [["A", one, "B"]])
        check("wrap_return set type", any(isinstance(x, InlineStr) for x in set_out), True)
        frozen_out = wrap_return(frozenset(["A" + k + "B"]), {"a": a})
        check("wrap_return frozenset", [x.to_list() for x in frozen_out], [["A", one, "B"]])

        # -- Aha-style signatures / class subscription --------------------
        check("class_getitem", InlineStr[str] is InlineStr, True)
        out_map: dict = {}
        InlineStr.from_obj("x", out_map)
        check("from_obj positional mapping", out_map, {})
        seg_map: dict = {}
        InlineStr.from_obj(one, seg_map)
        check("from_obj positional mapping seg", list(seg_map.values()), [one])
        iter_map: dict = {}
        InlineStr.from_iterable(["x", one], iter_map)
        check("from_iterable positional mapping", list(iter_map.values()), [one])

        # -- decorator must preserve the full generator protocol ----------
        @inline_aware
        def dec_gen(a):
            got = yield "ready"
            yield got + a

        g = dec_gen(a)
        check("decorator gen first", next(g).to_list(), ["ready"])
        check("decorator gen send", g.send("S").to_list(), ["SA", one, "B"])

        @inline_aware
        def dec_gen_throw(a):
            try:
                yield "ready"
            except ValueError:
                yield a + "!"

        g = dec_gen_throw(a)
        next(g)
        check("decorator gen throw", g.throw(ValueError).to_list(), ["A", one, "B!"])

        @inline_aware
        def dec_gen_return(a):
            yield "x"
            return "A" + k + "B"

        g = dec_gen_return(a)
        next(g)
        try:
            next(g)
        except StopIteration as exc:
            check("decorator gen return", exc.value.to_list(), ["A", one, "B"])
        else:
            raise AssertionError("decorator generator did not stop")

        @inline_aware
        async def dec_async_gen(a):
            got = yield "ready"
            yield got + a

        async def run_async_send():
            g = dec_async_gen(a)
            first = await g.__anext__()
            second = await g.asend("S")
            try:
                await g.__anext__()
            except StopAsyncIteration:
                pass
            return first, second

        first, second = asyncio.run(run_async_send())
        check("decorator async gen first", first.to_list(), ["ready"])
        check("decorator async gen asend", second.to_list(), ["SA", one, "B"])

        @inline_aware
        async def dec_async_gen_throw(a):
            try:
                yield "ready"
            except ValueError:
                yield a + "!"

        async def run_async_throw():
            g = dec_async_gen_throw(a)
            await g.__anext__()
            return await g.athrow(ValueError)

        check(
            "decorator async gen athrow",
            asyncio.run(run_async_throw()).to_list(),
            ["A", one, "B!"],
        )

        # -- re.Match groups ----------------------------------------------
        import re

        check("regex group auto", m.regex_group(a).to_list(), [one, "B"])
        check("regex group chain auto", m.regex_group_chain(a).to_list(), ["A", one, "b"])

        check(
            "internal conflict",
            m.internal_conflict(a).to_list(),
            ["A", one, "BC", m.OTHER, "D"],
        )

        check("group inside", m.group_inside(a), ["InlineStr", [one, "B"]])
        check("groups inside", m.groups_inside(a), [[one, "B"], [one, "B"]])
        check("unrelated .group", m.unrelated_group(), "plain:3")

        # -- manually configured method adapter ---------------------------
        check("custom method adapter", m.use_take(a).to_list(), ["A", one])
        check("custom method adapter fallback", m.use_take_plain(), "pl")

        # -- blanket call hooking (rewrite_calls=True) --------------------
        check("all-call adapter", m3.run(a), "A?")
        check("all-call nested", m3.run_nested(a), "helper:A?")
        check("all-call method adapter", m3.run_method(a).to_list(), ["A"])
        check("all-call method fallback", m3.run_method("plain"), "plain!")
        check("all-call wildcard", {"Holder", "combine"} <= set(_seen_calls), True)

        # -- plain group loses the mapping; explicit adapters restore it --
        a = InlineStr(f"A{k}B", {k: one})
        match = re.search(r"(A)(.B)", a)
        check("plain group loses mapping", isinstance(match.group(0), InlineStr), False)
        check("inline_group", inline_group(match, a, 0).to_list(), ["A", one, "B"])
        check("inline_group 1", inline_group(match, a, 1).to_list(), ["A"])
        check("inline_groups", [g.to_list() for g in inline_groups(match, a)], [["A"], [one, "B"]])
        proxy = inline_match(match, a)
        check("inline_match group", proxy.group(0).to_list(), ["A", one, "B"])
        check("inline_match default source", inline_match(match).group(0).to_list(), ["A", one, "B"])
        check("inline_match multi", [g.to_list() for g in proxy.group(2, 1)], [[one, "B"], ["A"]])
        check("inline_match span", proxy.span(1), (0, 1))
        optional = inline_match(re.search(r"(A)(X)?", a), a)
        check("inline_match missing group", optional.group(2), None)
        check("inline_match groups default", optional.groups("d"), ("A", "d"))
        check(
            "inline_findall",
            [x.to_list() for x in inline_findall(r"A.B", a)],
            [["A", one, "B"]],
        )

        print("all AST tests passed")
    finally:
        sys.path.remove(root)
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
