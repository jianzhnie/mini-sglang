---
paths:
  - "**/*.py"
  - "**/*.pyi"
---

# Python Style Guide

Based on the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html), adapted for modern Python 3.10+.

Recommended tooling: ruff, black (88 chars, double quotes), isort, mypy.

---

## 1 Module Mechanics

### `__all__` (recommended)

Recommended for public modules: export an explicit `__all__` list as the source of truth for the module's public API. Not required — project does not use wildcard imports, and missing `__all__` has no functional impact.

```python
__all__ = ["UserService", "create_user", "UserError"]
```

### `from __future__ import annotations` (optional)

Enables PEP 604 (`X | Y`) and lazy evaluation of annotations. Note that this project targets Python 3.10+, so `X | Y` syntax works natively without the import.

**Skip this import if your code uses** `torch.jit.script`, runtime `typing.get_type_hints()`, or `ClassVar` / `InitVar` in dataclasses — lazy string annotations can break these at runtime. Otherwise it is safe to include.

### Main guard

All executable scripts define and call a `main()` function:

```python
def main() -> None:
    ...

if __name__ == "__main__":
    main()
```

### Import ordering

Groups (alphabetical within each, one blank line between):

1. `from __future__ import ...` (if present)
2. Standard library
3. Third-party
4. First-party / local

Rules:
- `import x` for packages; `from x import y` where `x` is the package prefix
- `import y as z` only for widely-recognized abbreviations (`np`, `pd`)
- No relative imports, no wildcard imports, no multiple imports per line

```python
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import ClassVar, Final, TypeAlias

import numpy as np
import pandas as pd

from myproject.core import Config
from myproject.utils import helper
```

Prefer `collections.abc` over `typing` for abstract base types: `from collections.abc import Iterator` not `from typing import Iterator`.

---

## 2 Formatting

**Line length**: Maximum 88 characters. Use implicit line joining inside `()`, `[]`, `{}`. Never use `\` backslash continuation.

**Indentation**: 4 spaces. No tabs.

**Blank lines**: 2 blank lines between top-level definitions (class, function). 1 blank line between method definitions. 1 blank line between import groups.

**Whitespace**: No spaces inside `()`, `[]`, `{}`. One space after `,`, `:`, `;`. One space on both sides of binary operators (`+`, `-`, `==`, etc.). No spaces around `=` in keyword arguments: `func(a=1)`. One space around `=` in type-annotated defaults: `def func(a: int = 1)`.

**Trailing comma**: Required when closing bracket is on its own line:

```python
my_list = [
    1,
    2,
    3,
]
```

## 3 Naming

| Type | Convention | Example |
|------|------------|---------|
| Module / package | `lower_snake_case` | `my_module` |
| Class / exception / type alias | `CapWords` | `MyClass`, `ValueError`, `ProcessorConfig` |
| Function / method / variable | `lower_snake_case` | `my_function`, `block_size` |
| Constant | `UPPER_SNAKE_CASE` | `MAX_SIZE` |
| Private (internal) | leading underscore | `_internal_func` |

Prohibited:
- Hyphens, names carrying type info (`id_to_name_dict`), dunder names (reserved), single-char names except in tight loops (`i`, `j`, `k`)

---

## 4 Type Annotations

### Required

All public functions and methods must annotate parameters and return type.

### Modern syntax (3.10+)

- `X | Y` instead of `Union[X, Y]`
- `X | None` instead of `Optional[X]`
- Built-in generics: `list[int]`, `dict[str, int]`, `tuple[int, str]`

### Special types

| Type | When |
|------|------|
| `ClassVar` | Class-level mutable attributes |
| `TypeAlias` | Complex type shorthand at module level |
| `Final` | Constants that should not be reassigned |
| `Any` | Only when the type truly cannot be expressed |

### None defaults

Explicit `None` default requires `| None` in the type:

```python
def func(a: str | None = None) -> None:  # correct
    ...
```
```python
def func(a: str = None) -> None:          # wrong
    ...
```

### Rules

- `self` and `cls` need not be annotated
- `__init__` return type need not be annotated
- Favor `collections.abc` for abstract types: `Sequence`, `Mapping`, `Iterable` over their `typing` equivalents

---

## 5 Docstrings

Always `"""` triple double-quotes. Follow the structure below for public API.

### Module

```python
"""One-line summary of the module.

Leave one blank line, then describe the module in more detail.
"""
```

### Function / Method

```python
def my_function(arg1: int, arg2: str) -> bool:
    """One-line summary, no more than 88 chars, ending with a period.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When arg1 is negative.
    """
```

Rules:
- Summary: one line, ends with period / question mark / exclamation
- `Args`, `Returns`, `Raises` sections for public API
- Private methods (`_`) may use a one-line docstring or omit it
- Do not document the obvious

### Class

```python
class MyClass:
    """One-line summary.

    Attributes:
        attr1: Description of attr1.
        attr2: Description of attr2.
    """
```

---

## 6 Comments

- Explain **WHY**, not **WHAT**
- Block comments (`#`) above the described code, not inline
- Delete commented-out code before committing
- Do not reference PR numbers or ticket IDs in code comments

TODO format:

```python
# TODO(username): What needs to be done.
```

---

## 7 Error Handling

- Catch the most specific exception; no bare `except:` or `except Exception:` unless re-raising immediately
- Never silently swallow exceptions — log at minimum
- Custom exceptions must inherit an existing exception and end with `Error`
- Use `raise from` to preserve tracebacks:

```python
try:
    data = load_file(path)
except FileNotFoundError as e:
    raise RuntimeError(f"Failed to load {path}") from e
```

- Use `with` for files and resources; use `finally` for cleanup
- Minimize code inside `try` blocks — only the line that can fail

### assert

Use `assert` only for internal invariants (logic errors, not recoverable). Never for input validation or runtime checks — assertions can be disabled with `-O`.

```python
# Yes — internal invariant
assert x >= 0, "internal: negative value after normalization"

# No — runtime check
assert os.path.exists(path), "Path must exist"
```

---

## 8 Values & Comparisons

### Boolean

| Prefer | Avoid |
|--------|-------|
| `if seq:` | `if len(seq) != 0:` |
| `if not seq:` | `if len(seq) == 0:` |
| `if val is None:` | `if val == None:` |
| `if flag:` | `if flag == True:` |
| `if not flag:` | `if flag == False:` |

When `0` is a valid value, compare explicitly: `if x == 0:` not `if not x:`.

Use `isinstance` checks, not type identity: `isinstance(x, MyClass)` not `type(x) is MyClass`.

### Default arguments

Never use mutable objects as defaults:

```python
def foo(a, b: list[int] | None = None):
    if b is None:
        b = []
```
```python
def foo(a, b: Sequence[int] = ()):  # also valid — immutable tuple
    ...
```

### String formatting

- Prefer f-strings for inline formatting
- Never use `+` or `+=` for string accumulation in loops — use `"".join()` or `io.StringIO`
- Logging: use `%`-style placeholders for lazy evaluation:

```python
logger.info("Loading from %s", path)     # correct
logger.info(f"Loading from {path}")       # wrong — eager eval
```

Module-level logger pattern:

```python
import logging

logger = logging.getLogger(__name__)
```

---

## 9 Patterns

### Comprehensions & generators

Allowed for simple cases. No multiple `for` clauses:

```python
result = [x * 2 for x in data if x > 0]     # correct
has_positive = any(x > 0 for x in data)      # correct — generator, not list
has_positive = any([x > 0 for x in data])    # wrong — unnecessary list
```

### Dataclasses

Prefer `@dataclass` for data containers. Use `__post_init__` for validation that runs after field initialization:

```python
@dataclass
class Config:
    name: str
    max_items: int = 100

    def __post_init__(self) -> None:
        if self.max_items < 1:
            raise ValueError(f"max_items must be >= 1, got {self.max_items}")
```

### Enums

Use `Enum` + `auto()` for enumerations:

```python
from enum import Enum, auto

class Status(Enum):
    PENDING = auto()
    ACTIVE = auto()
    COMPLETED = auto()
```

### Properties

Use `@property` instead of getter/setter methods:

```python
@property
def full_name(self) -> str:
    return f"{self.first} {self.last}"
```

### Match / case (3.10+)

Use pattern matching for complex branching on enums, dataclasses, or tuples. Prefer over long `if`/`elif` chains.

### Lambdas

Only for trivial one-liners (e.g. sort keys). Use a nested `def` for anything multi-line.

### Context managers

Use `with` for any resource with clean-up. When defining custom context managers, prefer `@contextmanager` for simple cases, `__enter__`/`__exit__` for stateful ones.

### Classes

- Instance attributes initialized in `__init__`, not dynamically added later
- Avoid `staticmethod` — use module-level functions
- `classmethod` only for named constructors or class-level state
- Inherit from `object` only in Python 2 compat code

---

## 10 General Conventions

- Use `pathlib.Path` over `os.path` for file operations
- Use `logging` module, never `print()`, for diagnostic output
- Keep functions focused; consider splitting when exceeding ~40 lines
- Be consistent with existing code in the same file
