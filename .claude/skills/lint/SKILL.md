---
name: lint
description: Run shellcheck static analysis on shell scripts. Usage: /lint [file_or_dir] — defaults to all .sh files under scripts/, tools/, and examples/.
---

Run shellcheck on the specified target.

If the user provides a file path, run shellcheck on that file only:
```bash
shellcheck "$TARGET"
```

If the user provides a directory, run:
```bash
shellcheck "$TARGET"/**/*.sh
```

If no target is specified, run on the full project:
```bash
shellcheck scripts/**/*.sh tools/*.sh examples/*.sh
```

Report findings in a concise summary: file, line, severity (error/warning/info), and message. If no issues, confirm the target is clean.
