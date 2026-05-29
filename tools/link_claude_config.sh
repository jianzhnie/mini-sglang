#!/bin/bash
# link_claude_config.sh — Symlink shared .claude config from oh-my-claude-code to a project.
#
# Usage: link_claude_config.sh [TARGET_PROJECT_DIR]
#   TARGET_PROJECT_DIR: path to the project that should reuse the shared config
#                       (defaults to current directory)

set -euo pipefail

readonly SOURCE_BASE="/Users/robin/work_dir/oh-my-claude-code/.claude"
readonly TARGET_DIR="${1:-$(pwd)}"
readonly TARGET_BASE="${TARGET_DIR}/.claude"

# Directories and files to symlink from oh-my-claude-code
readonly SHARED_DIRS=(
    agents
    skills
    hooks
    rules
    output-styles
    commands
)

readonly SHARED_FILES=(
    statusline.sh
    .mcp.json
)

usage() {
    echo "Usage: $(basename "$0") [TARGET_PROJECT_DIR]"
    echo ""
    echo "Symlinks shared .claude/ config (agents, skills, hooks, rules, etc.)"
    echo "from oh-my-claude-code into the target project."
    echo ""
    echo "Project-specific files (CLAUDE.md, settings.json, .env, memory.md)"
    echo "are left untouched."
    exit 1
}

err() {
    echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
}

die() {
    err "$*"
    exit 1
}

main() {
    [[ -d "$SOURCE_BASE" ]] || die "Source config not found: ${SOURCE_BASE}"
    [[ -d "$TARGET_DIR" ]] || die "Target project dir not found: ${TARGET_DIR}"

    mkdir -p "$TARGET_BASE"

    echo "Linking shared config from oh-my-claude-code → ${TARGET_DIR}"
    echo ""

    for dir in "${SHARED_DIRS[@]}"; do
        printf "  %-20s" "${dir}/"
        rm -rf "${TARGET_BASE:?}/${dir}"
        ln -s "${SOURCE_BASE}/${dir}" "${TARGET_BASE}/${dir}"
        echo "✓"
    done

    for file in "${SHARED_FILES[@]}"; do
        printf "  %-20s" "${file}"
        rm -f "${TARGET_BASE:?}/${file}"
        ln -s "${SOURCE_BASE}/${file}" "${TARGET_BASE}/${file}"
        echo "✓"
    done

    echo ""
    echo "Done. These remain project-specific: CLAUDE.md, settings.json, .env, memory.md"
}

main "$@"
