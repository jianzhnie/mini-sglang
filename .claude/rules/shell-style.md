---
paths:
  - "**/*.sh"
  - "**/*.bash"
---

# Shell Script Style

All shell scripts in this project must follow these conventions. Based on Google Shell Style Guide, adapted for bash 4.2+ compatibility.

## Table of Contents

- [File Structure](#file-structure)
- [Naming Conventions](#naming-conventions)
- [Comments](#comments)
- [Formatting](#formatting)
- [Variables and Constants](#variables-and-constants)
- [Functions](#functions)
- [Conditionals and Testing](#conditionals-and-testing)
- [Error Handling](#error-handling)
- [Command Substitution and Pipelines](#command-substitution-and-pipelines)
- [Size Limits](#size-limits)
- [Lint](#lint)
- [Best Practices](#best-practices)

## File Structure

Scripts must be organized in this order:

1. **Shebang** and `set` options
2. **Constants** and `readonly` exports
3. **Functions** (all together, no executable code between functions)
4. **`main` function** (for scripts with ≥1 other function)
5. **Call to `main`** at the bottom: `main "$@"`

```bash
#!/bin/bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
readonly VERSION="1.0"

my_func() {
    ...
}

main() {
    my_func
}

main "$@"
```

Rules:

- **Shebang**: `#!/bin/bash` (not `#!/usr/bin/env bash` — for bash 4.2+ compatibility)
- **Directly-executed scripts** must start with `set -euo pipefail`
- **Sourced files** (e.g. `common.sh`) must NOT set shell options — they inherit from the caller
- **Library files** must use `.sh` suffix; executable scripts may omit suffix

## Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Script / file | `lower_snake_case` | `run_server.sh`, `common.sh` |
| Function | `lower_snake_case` | `run_server()`, `mylib::helper()` |
| Variable | `lower_snake_case` | `local count`, `file_path` |
| Constant / `readonly` | `UPPER_SNAKE_CASE` | `readonly MAX_RETRIES=3` |
| Exported variable | `UPPER_SNAKE_CASE` | `export PATH_TO_DATA` |
| Loop variable | match collection name | `for file in "${files[@]}"; do` |

Rules:

- Use `readonly` or `declare -r` to declare constants explicitly
- Use `local` for function-internal variables
- Associated variables should share naming style: `user_name`, `user_id`, `user_home`

## Comments

### File header

Every executable script must have a header comment describing purpose and usage:

```bash
#!/bin/bash
# run_server.sh — Starts the inference server.
#
# Usage: run_server.sh [--port PORT] [--workers N]
# Options:
#   --port     Server port (default: 8000)
#   --workers  Number of worker processes (default: 4)
```

### Function comments

Functions should have a header describing purpose, arguments, and return value:

```bash
# Runs health check on the specified endpoint.
# Args:
#   $1: endpoint URL
#   $2: timeout in seconds (default: 30)
# Returns:
#   0 if healthy, 1 otherwise
health_check() {
    ...
}
```

### Inline comments

- Explain **why**, not what — the code itself should show what
- Use `# TODO(username): description` for future work
- Use UTF-8 encoding; prefer English to avoid encoding issues

## Formatting

- **Indentation**: 4 spaces, never tabs
- **Line length**: max 120 characters (break lines with `\` or grouping)
- **Trailing semicolons**: omit `;` at end of lines
- **Trailing commas**: when items span multiple lines, add trailing comma on last item

### Control structure formatting

```bash
# Correct — ; do / ; then on same line as for/if/while
for file in "${files[@]}"; do
    echo "$file"
done

if [[ -f "$config" ]]; then
    source "$config"
elif [[ -d "$config_dir" ]]; then
    source "${config_dir}/default.conf"
else
    echo "Config not found" >&2
    exit 1
fi

# Correct — case formatting
case "$action" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    *)
        echo "Unknown action: $action" >&2
        exit 1
        ;;
esac
```

Rules:

- `; do` and `; then` must be on the same line as `for`/`if`/`while`
- `else`, `fi`, `done` on their own line
- Do not mix single-line and multi-line `if` styles in the same script

### Quoting

- **Always double-quote** strings containing variables, command substitutions, spaces, or shell metacharacters
- Use `"${var}"` (with braces) for clarity
- Do not quote literal integers
- Do not quote command options or path literals without variables

```bash
# Correct
echo "Processing: ${file_name}"
cmd --max="${MAX_COUNT}"

# Wrong — unquoted variable
echo $file_name

# Correct — literal integer needs no quotes
exit 1
```

## Variables and Constants

### Declaration

```bash
# Correct — separate declaration and assignment for command substitution
local file_count
file_count=$(wc -l < "$file")

# Correct — simple assignment on one line
local name="server"
readonly MAX_CONN=100

# Wrong — command substitution in same line as declaration
local file_count=$(wc -l < "$file")
```

### Defaults and substitutions

```bash
# Use :- for default values
port="${SERVER_PORT:-8000}"
name="${1:?Error: server name is required}"
```

### Arrays

- Use arrays for lists of items; never use strings for sequences
- Always quote array expansions

```bash
local files=("config.json" "model.pt" "tokenizer.json")
for f in "${files[@]}"; do
    echo "$f"
done
```

### Special variables

- Always quote `$@` and `$*`: `"$@"` (preferred) or `"$*"`
- Prefer `"$@"` over `"$*"` unless you explicitly need concatenation

## Functions

### Definition

```bash
# Correct
run_server() {
    ...
}

# Correct — library functions
mylib::helper() {
    ...
}
```

Rules:

- Opening brace `{` must be on the same line as the function name
- No space between function name and `()`
- `function` keyword is optional; use it consistently if chosen

### Return values

- Return **exit status only** (0 = success, non-zero = failure)
- Do not use `return` to pass computed results — use `echo` or a variable

```bash
# Correct — return status
is_file_readable() {
    local file="$1"
    [[ -r "$file" ]]
}

# Correct — output result via echo
get_file_size() {
    local file="$1"
    stat -c %s "$file"
}

# Wrong — returning computed result
compute_sum() {
    local a="$1" b="$2"
    return $((a + b))  # Incorrect: return expects 0-255 exit code
}
```

### Local variables

```bash
my_func() {
    local input_file="$1"
    local count=0
    ...
}
```

## Conditionals and Testing

| Preferred | Avoid |
|-----------|-------|
| `[[ ... ]]` | `[ ... ]`, `test`, `/usr/bin/[` |
| `(( ... ))` for numeric | `let`, `$[ ... ]`, `expr` |
| `-z`, `-n` for empty strings | `"${var}X"` tricks |
| `==` for string equality | `=` (confuses with assignment) |

```bash
# Correct
if [[ -z "$name" ]]; then
    echo "Name is empty"
fi

if (( retries > MAX_RETRIES )); then
    echo "Too many retries"
fi

# Correct — explicit -n / -z
if [[ -n "$value" ]]; then ... fi
if [[ -z "$value" ]]; then ... fi
```

## Error Handling

### Exit on error

Always use `set -euo pipefail` at the top of executable scripts:

- `set -e` — exit immediately on command failure
- `set -u` — exit on undefined variable
- `set -o pipefail` — pipeline fails if any command fails

### Error messages

- Write all error messages to **STDERR** (`>&2`)
- Use a helper function for consistent error output

```bash
err() {
    echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
}

die() {
    err "$*"
    exit 1
}

# Usage
die "Failed to load config: ${config_file}"
```

### Checking results

- Check return values explicitly when not using `set -e`
- Use `PIPESTATUS` immediately after pipelines when not using `pipefail`

```bash
some_command
if [[ $? -ne 0 ]]; then
    die "Command failed"
fi
```

### Cleanup

Use `trap` for cleanup on exit or error:

```bash
cleanup() {
    rm -f "${temp_file}"
}
trap cleanup EXIT
```

## Command Substitution and Pipelines

### Command substitution

- Always use `$(command)` — never backticks
- Declare and assign on separate lines (see Variables section)

### Pipelines

- Prefer built-in commands over external commands
- Avoid piping to `while read` — use process substitution instead

```bash
# Wrong — subshell prevents variable changes
ls | while read -r file; do
    count=$((count + 1))
done

# Correct — process substitution
while read -r file; do
    count=$((count + 1))
done < <(ls)
```

### Wildcards

- Always use explicit path with wildcards: `./*` not `*`

```bash
# Correct
rm -rf ./*

# Wrong
rm -rf *
```

## Size Limits

- **Scripts**: under 400 lines
- **Functions**: under 50 lines
- If a function exceeds 50 lines, extract helper functions
- If a script exceeds 400 lines, consider splitting into modules

## Lint

- All code must pass `shellcheck`
- Only `disable=SC2086` is allowed, and only with a justification comment

```bash
# shellcheck disable=SC2086 — word splitting is intentional here
rm -rf ${files_to_remove}
```

## Best Practices

- **Avoid `eval`** — it introduces security risks; use arrays or functions instead
- **Avoid aliases** in scripts — use functions
- **Avoid temporary files** — prefer process substitution; if needed, use `mktemp`
- **Avoid magic numbers** — define constants at the top
- **Avoid `cat | grep`** — use `grep pattern file` directly
- **Idempotency** — scripts should be safe to run multiple times
- **Absolute paths** in environment variable assignments; relative paths must use `./` prefix
- **Script directory**: get it reliably with `SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"`
- **Parameter validation**: check argument count or provide `--help`/`usage`

```bash
usage() {
    echo "Usage: $0 <config_file>"
    exit 1
}

[[ $# -eq 1 ]] || usage
```
