---
name: check
description: Run the full pre-commit check suite (shellcheck, flake8, isort, yapf, trailing-whitespace, etc.). Equivalent to pre-commit run --all-files.
---

Run the complete pre-commit check suite for the project:

```bash
pre-commit run --all-files
```

Report each hook result: PASS or FAIL with details. If any hook fails, list the affected files and the specific issues so the user can fix them.
