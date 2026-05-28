---
name: shell-check
description: Validate shell scripts with both syntax check (bash -n) and static analysis (shellcheck). Usage: /shell-check [file_or_dir] — defaults to all .sh files under scripts/, tools/, and examples/.
---

Run both bash syntax validation and shellcheck static analysis on the specified target.

## If the user provides a file path

Run both checks on that single file:

```bash
bash -n "$TARGET" 2>&1
shellcheck "$TARGET" 2>&1
```

## If the user provides a directory

Run both checks on all `.sh` files in that directory:

```bash
find "$TARGET" -name '*.sh' -exec bash -n {} \; 2>&1
shellcheck "$TARGET"/**/*.sh 2>&1
```

## If no target is specified

Run both checks on the full project's shell scripts:

```bash
find scripts/ tools/ examples/ -name '*.sh' -exec bash -n {} \; 2>&1
shellcheck scripts/**/*.sh tools/*.sh examples/*.sh 2>&1
```

## Reporting

For each file, report:
- **Syntax check** (`bash -n`): PASS or FAIL with parse error details
- **Static analysis** (`shellcheck`): PASS or clean, or list issues with file, line, severity (error/warning/info), and message

Give a final summary: total files checked, syntax pass/fail count, shellcheck clean/issue count.
