---
name: check-syntax
description: Validate bash syntax of shell scripts using bash -n. Usage: /check-syntax [file_path] — defaults to all .sh files under scripts/, tools/, and examples/.
---

Run bash syntax check on the specified target.

If the user provides a file path, validate that single file:
```bash
bash -n "$TARGET"
```

If no target is specified, validate all shell scripts in the project:
```bash
find scripts/ tools/ examples/ -name '*.sh' -exec bash -n {} \; 2>&1
```

Report results per file: PASS or FAIL with the error message. Give a final count of passed/failed files.
