"""Built-in smoke test: ``python -m aware_inline_str``."""

from . import InlineStr


class Seg:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Seg({self.name!r})"


def main() -> None:
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


if __name__ == "__main__":
    main()
