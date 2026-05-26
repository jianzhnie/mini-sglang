# Shell Script Style

All shell scripts in this project must follow these conventions.

## Script structure

- Directly-executed scripts must start with `set -euo pipefail`
- Sourced files (e.g. `common.sh`) must NOT set shell options — they inherit from the caller
- Shebang: `#!/bin/bash` (not `#!/usr/bin/env bash` — for bash 4.2+ compatibility)

## Syntax and formatting

- Double-quote all variable references: `"$var"`, `"${array[@]}"`
- Use `[[ ]]` for conditionals, not `[ ]` or `test`
- Use `$(command)` for substitution, not backticks
- `local` for function-scoped variables, `readonly` for constants
- 4-space indentation, max 120 char line width

## Size limits

- Scripts under 400 lines
- Functions under 50 lines
- If a function exceeds 50 lines, extract helper functions

## Lint

- All code must pass `shellcheck`
- Only `disable=SC2086` is allowed, and only with a justification comment
