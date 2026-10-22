# aware_inline_str

一个可以自动携带“非文本消息段”的 `str` 子类。

`InlineStr` 是普通 `str`，所以能直接参与所有字符串操作；每个实例额外保存
一个 `char -> object` 映射，用私有区（PUA）字符占位。字符串操作结束后，
`to_list()` 可以把文本和原始对象重新还原成列表。

与基于 `ContextVar` 的实现不同，映射保存在**每个实例**上，因此重入、线程、
异步任务和同一条命令的重复执行都不会互相污染。

## 文件

| 文件 | 说明 |
| --- | --- |
| `aware_inline_str.py` | 完整实现，单文件、无第三方依赖 |
| `test_core.py` | 类级别测试（不需要 AST hook） |
| `test_ast.py` | AST hook / 装饰器 / 适配器测试（自包含，自动生成 fixture） |

要求 Python 3.10+。

## 快速开始

```python
from aware_inline_str import InlineStr


class At:
    def __init__(self, qq): self.qq = qq
    def __repr__(self): return f"At({self.qq})"


s = InlineStr.from_iterable(["你好 ", At(123), "，看这里"])
print(s)                 # 你好 <PUA>，看这里
print(s.to_list())       # ['你好 ', At(123), '，看这里']

print((s + "！").to_list())        # ['你好 ', At(123), '，看这里！']
print(s.upper().to_list())         # ['你好 ', At(123), '，看这里']（At 不受影响）
print(s.split("，")[1].to_list())  # [At(123), '看这里']

# 也可以直接构造：value 中必须包含 mapping 的每个 key 字符
one = InlineStr("AB", {"": At(123)})
print(one.to_list())     # ['A', At(123), 'B']
```

`from_obj(obj)` 把单个对象包成一个 InlineStr；普通字符串直接返回文本形式。

### 兼容 Aha 的旧签名

```python
mapping = {}
InlineStr.from_iterable(["x", obj], mapping)   # 位置参数即输出 mapping
InlineStr.from_obj(obj, mapping)
InlineStr[str] is InlineStr                    # 注解写法
InlineStr.clear_map()                          # 兼容 no-op
```

## 自动传播范围

以下操作都会保留映射：

- 运算符：`+`、`*`、`%`、索引、切片、迭代
- 返回 `str` / `list[str]` / `tuple[str, ...]` 的字符串方法：
  `upper`、`lower`、`title`、`strip`、`replace`、`split`、`rsplit`、
  `partition`、`splitlines`、`format`、`format_map`、`join`、`translate` 等
- `str.format` / `%` 格式化、f-string 中的单字段替换
- `copy` / `deepcopy` / `pickle`
- 在函数边界（见下文）上，普通 `str` 结果以及 list/tuple/dict/set/
  namedtuple/dataclass/`SimpleNamespace` 容器会被递归恢复

## 冲突处理

两个 InlineStr 用同一个字符表示**不同对象**时，必须有一方换用新的 PUA
字符。实现有三层保护：

1. **构造期注册表**（默认开启）：全局记录“某个字符当前属于哪个对象”，
   构造时发现冲突立即重键。函数内部新建的值也能覆盖。
2. **函数入口 `_prepare`**：参数之间、参数与调用方 locals（AST hook）或
   闭包/默认值（`inline_aware`）之间再次检查冲突。
3. **字面量碰撞**：如果 key 字符同时出现在普通字符串参数里，InlineStr
   一方换 key，避免把普通文本误认为消息段。

字符从全部 137,468 个私有区/未分配码位中顺序分配，不是随机碰撞。
如果确定不需要构造期检查，可以关闭：

```python
InlineStr.configure(key_registry=False)
```

## 函数边界

字符串在 C 层被拼成普通 `str` 后（`"".join(...)`、`re.sub`、`json.dumps`、
`os.path.join` 等），映射会暂时丢失，需要在函数出口恢复。有三种方式。

### 1. `@inline_aware`

```python
from aware_inline_str import InlineStr, inline_aware

@inline_aware
def render(user, seg):
    return "".join([user, " -> ", seg])

render(a, obj).to_list()   # 参数和闭包里的映射都会参与恢复
```

- 入口：参数冲突重键（包括闭包与默认值）
- 出口：把普通 `str`/容器恢复成 InlineStr
- 支持普通函数、`async` 函数、生成器、异步生成器
- 生成器的 `send` / `throw` / `close`、`yield from` 的返回值、
  异步生成器的 `asend` / `athrow` / `aclose` 都保持原语义

### 2. AST import hook

在导入目标模块**之前**安装：

```python
from aware_inline_str import install_ast_hook

install_ast_hook(__name__, "my_package")   # runtime 必须是本模块的导入名
import my_package.mod
```

hook 会改写目标模块的 Python 源码：

- 函数入口：`prepare_many(...)` 对参数重键
- 每个 `return` / `yield` / `yield from`：用当前 locals 恢复映射
- lambda：整体包一层 `inline_aware`
- 可选：改写调用点，走运行时可配置的适配器（见下）

`prefixes` 传 `"*"`（或不传）会处理之后导入的所有纯 Python 模块，适合受控
进程，不建议在生产环境全局开启。

### 3. `patch_module`

对已经导入、或无法被 AST 改写的模块，在 Python 调用边界包装：

```python
from aware_inline_str import patch_module
import some_module

patch_module(some_module)
```

会处理模块内公开函数和公开方法（含 `property` / `staticmethod` /
`classmethod`）。它看不到函数内部，因此函数内部凭空创建的 mapping 无法恢复；
这种情况仍需要 hook 或 `@inline_aware`。

## 调用点适配器

AST hook 的方法/函数改写不是硬编码的，而是由可注册的适配器决定。

```python
from aware_inline_str import (
    DEFAULT_MATCH_METHODS, NOT_HANDLED,
    register_call_adapter, register_method_adapter, install_ast_hook,
)


def take_adapter(obj, name, args, kwargs):
    value = getattr(obj, "value", None)
    if isinstance(value, InlineStr):
        return value[: args[0]]      # 返回替换结果
    return NOT_HANDLED               # 不处理，继续调用真实方法


register_method_adapter("take", take_adapter)

install_ast_hook(
    __name__, "my_package",
    rewrite_methods=DEFAULT_MATCH_METHODS | {"take"},
)
```

- 方法适配器：`adapter(obj, name, args, kwargs)`
- 函数适配器：`adapter(func, name, args, kwargs)`，按 `func.__name__` 匹配
- 注册名字 `"*"` 表示匹配所有改写过的调用
- 返回 `NOT_HANDLED` 表示放行；否则返回值即调用结果
- `unregister_method_adapter(name[, adapter])` /
  `unregister_call_adapter(name[, adapter])` 可注销

改写强度：

| 参数 | 效果 |
| --- | --- |
| `rewrite_methods=False` | 不改写任何方法调用 |
| `rewrite_methods=True` | 所有 `obj.name(...)` 都走 dispatcher |
| `rewrite_methods={...}` | 只改写列出的方法名（默认 `group`/`groups`/`groupdict`） |
| `rewrite_calls=True` | 所有 `f(...)` 以及未选中的方法调用都走 dispatcher |

`rewrite_calls=True` 会给目标模块的每条调用增加一次 Python 分发，语义不变，
但只建议在调试或受控场景使用。

## `re.Match`

`re.Match` 是 C 不可变类型，不能继承或 monkeypatch，`m.group()` 返回的是
普通 `str` 拷贝。两种恢复方式：

**被 hook 覆盖的代码**：默认适配器会自动处理 `.group/.groups/.groupdict`，
函数内部拿到的就是 InlineStr，不需要等到 return：

```python
def f(a):
    m = re.match(r"(.)(.*)", a)
    return m.group(2)        # 已是 InlineStr
```

**未被 hook 覆盖的代码**：使用显式适配器；组是源串的连续切片，因此直接从
原 InlineStr 切片即可精确恢复：

```python
from aware_inline_str import (
    inline_match, inline_group, inline_groups, inline_findall, inline_finditer,
)

m = re.search(r"(A)(.B)", s)
inline_group(m, s, 0)
inline_groups(m, s)
inline_match(m, s).group(2)
inline_findall(r"A.B", s)
```

## 配置

```python
InlineStr.configure(
    text_types=(MyText,),   # 额外的“文本类”类型
    scan_objects=True,      # 恢复时遍历普通对象的 vars(obj)
    key_registry=False,     # 关闭构造期冲突检查
    error_type=MyError,     # 码位耗尽时抛出的异常
)
```

`scan_objects=True` 可以恢复普通对象属性里的 InlineStr，但大对象图会更慢。

## 已知限制

- 改变或删除 key 字符本身的转换会破坏来源：`encode`、base64、压缩、
  加密、`json.dumps(ensure_ascii=True)`、替换 PUA 字符的正则等。
- 持久化（数据库、缓存、文件）不保留 mapping，除非使用本类的 pickle 支持。
- 未安装 hook 且未 patch 的模块，如果内部执行 C 层字符串操作，mapping 会在
  函数内部丢失。
- `exec` / `eval` 生成的代码不会被改写。
- 普通文本恰好包含相同 PUA 字符时，只有双方都进入 `_prepare` 才能消歧；
  从任意 C 代码返回的字符串只能做启发式恢复。
- 两个已存在但 key 冲突的值如果绕过了所有保护（例如来自其他库），在普通
  `str` 边界上是真正有歧义的；恢复时会丢弃该 key 而不是错误归属。

## 运行测试

```bash
python test_core.py
python test_ast.py
```

`test_ast.py` 会在临时目录中生成带 hook 的 fixture 包，不需要额外文件。
