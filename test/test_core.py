"""Core (class-level) tests for aware_inline_str.

Everything here runs without the AST import hook: construction, operators,
string methods, conflict resolution, formatting, copy/pickle.

    python test/test_core.py
"""

from __future__ import annotations

import copy
import os
import pickle
import sys

# Run from a source checkout without installing (src layout).
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src",
    ),
)

from aware_inline_str import InlineStr


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


def main() -> None:
    one = Seg("one")
    two = Seg("two")
    k1 = ""
    k2 = ""

    a = InlineStr(f"A{k1}B", {k1: one})
    b = InlineStr(f"C{k2}D", {k2: two})

    # -- construction ------------------------------------------------------
    check("from_iterable", InlineStr.from_iterable(["x", one, "y"]).to_list(), ["x", one, "y"])
    check("from_obj text", InlineStr.from_obj("x").to_list(), ["x"])
    check("from_obj seg", InlineStr.from_obj(one).to_list(), [one])
    check("from_obj inline", InlineStr.from_obj(a).to_list(), ["A", one, "B"])

    # -- conflict re-keying ------------------------------------------------
    same_a = InlineStr(f"A{k1}B", {k1: one})
    same_b = InlineStr(f"C{k1}D", {k1: two})
    check("add conflict", (same_a + same_b).to_list(), ["A", one, "BC", two, "D"])
    check("radd conflict", ("X" + same_a).to_list(), ["XA", one, "B"])
    check("join conflict", InlineStr("|").join([same_a, same_b]).to_list(), ["A", one, "B|C", two, "D"])
    check(
        "format conflict",
        InlineStr("{} {}").format(same_a, same_b).to_list(),
        ["A", one, "B C", two, "D"],
    )
    check(
        "percent conflict",
        (InlineStr("%s %s") % (same_a, same_b)).to_list(),
        ["A", one, "B C", two, "D"],
    )

    # A literal collision keeps the plain text and re-keys the segment.
    lit = same_a + f"L{k1}M"
    check("literal collision", lit.to_list(), ["A", one, f"BL{k1}M"])
    assert lit.mapping and k1 not in lit.mapping

    # -- operators and string methods -------------------------------------
    check("upper", a.upper().to_list(), ["A", one, "B"])
    check("lower", a.lower().to_list(), ["a", one, "b"])
    check("casefold", a.casefold().to_list(), ["a", one, "b"])
    check("title", a.title().to_list(), ["A", one, "B"])
    check("swapcase", a.swapcase().to_list(), ["a", one, "b"])
    check("capitalize", a.capitalize().to_list(), ["A", one, "b"])
    check("center", a.center(9, ".").to_list(), ["...A", one, "B..."])
    check("ljust", a.ljust(5, ".").to_list(), ["A", one, "B.."])
    check("rjust", a.rjust(5, ".").to_list(), ["..A", one, "B"])
    check("zfill", a.zfill(5).to_list(), ["00A", one, "B"])
    check("strip", (" " + a + " ").strip().to_list(), ["A", one, "B"])
    check("lstrip", (" " + a).lstrip().to_list(), ["A", one, "B"])
    check("rstrip", (a + " ").rstrip().to_list(), ["A", one, "B"])
    check("removeprefix", ("X" + a).removeprefix("X").to_list(), ["A", one, "B"])
    check("removesuffix", (a + "X").removesuffix("X").to_list(), ["A", one, "B"])
    check("split", [p.to_list() for p in a.split("B")], [["A", one], [""]])
    check("rsplit", [p.to_list() for p in a.rsplit("A")], [[""], [one, "B"]])
    check("splitlines", [p.to_list() for p in (a + "\nC").splitlines()], [["A", one, "B"], ["C"]])
    check("partition", [p.to_list() for p in a.partition("B")], [["A", one], ["B"], [""]])
    check("rpartition", [p.to_list() for p in a.rpartition("A")], [[""], ["A"], [one, "B"]])
    check("replace", a.replace("A", "X").to_list(), ["X", one, "B"])
    check("replace new inline", a.replace("A", b).to_list(), ["C", two, "D", one, "B"])
    check("expandtabs", ("\t" + a).expandtabs(2).to_list(), ["  A", one, "B"])
    check("getitem char", a[1].to_list(), [one])
    check("getitem plain", a[0].to_list(), ["A"])
    check("getitem slice", a[0:2].to_list(), ["A", one])
    check("mul", (a * 2).to_list(), ["A", one, "BA", one, "B"])
    check("mul zero", (a * 0).to_list(), [""])
    check("rmul", (2 * a).to_list(), ["A", one, "BA", one, "B"])
    check("iter", [c.to_list() for c in a], [["A"], [one], ["B"]])

    # -- format / percent / format_map / translate ------------------------
    check("format", "{}".format(a).to_list(), ["A", one, "B"])
    check("format spec", format(a, ">5").to_list(), ["  A", one, "B"])
    check("format kwargs", InlineStr("{x}").format(x=a).to_list(), ["A", one, "B"])
    check("format_map", InlineStr("{x}").format_map({"x": a}).to_list(), ["A", one, "B"])
    check("percent mapping", (InlineStr("%(x)s") % {"x": a}).to_list(), ["A", one, "B"])
    check("rmod", ("pre%spost" % a).to_list(), ["preA", one, "Bpost"])
    table = str.maketrans({"A": "X"})
    check("translate", a.translate(table).to_list(), ["X", one, "B"])

    # -- to_list override, copy, pickle -----------------------------------
    check("to_list mapping", a.to_list({k1: two}), ["A", two, "B"])

    c = copy.copy(a)
    check("copy", c.to_list(), ["A", one, "B"])
    d = copy.deepcopy(a)
    check("deepcopy", d.to_list(), ["A", one, "B"])
    p = pickle.loads(pickle.dumps(a))
    check("pickle", p.to_list(), ["A", one, "B"])

    # The same object twice in one result must share a key.
    twice = InlineStr.from_iterable([a, "|", a])
    check("repeat object", twice.to_list(), ["A", one, "B|A", one, "B"])
    assert twice.mapping[k1] is one

    # -- configurable text types ------------------------------------------
    class T:
        def __init__(self, text):
            self.text = text

    old_types = InlineStr._text_types
    InlineStr.configure(text_types=(T,))
    try:
        check("text type", InlineStr.from_iterable([T("x"), one, T("y")]).to_list(), ["x", one, "y"])
    finally:
        InlineStr.configure(text_types=old_types)

    # Separator/chars mappings must not leak into results.
    sep = InlineStr("Z", {"Z": two})
    check("split sep mapping", [p.to_list() for p in a.split(sep)], [["A", one, "B"]])
    check("strip chars mapping", ("Z" + a + "Z").strip(sep).to_list(), ["A", one, "B"])

    print("all core tests passed")


if __name__ == "__main__":
    main()
