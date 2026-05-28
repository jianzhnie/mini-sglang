# Project Memory

This file records non-obvious decisions, patterns, and context that complement `CLAUDE.md`.
CLAUDE.md covers architecture and conventions; this file covers **why decisions were made**
and **what changed recently**.

## Design Decisions

### Hook Architecture: merged by trigger, not by function

Hooks are merged by trigger timing (PreToolUse, PostToolUse) rather than by concern
(security, formatting, linting). Rationale: each hook invocation costs a JSON parse +
subprocess spawn. Fewer hooks = less overhead.

- `pre-tool-use.sh`: dangerous command blocking + file protection (same PreToolUse trigger)
- `post-tool-use.sh`: auto-format + shellcheck (same Edit|Write trigger)

### Agents: generic, not project-specific

All agents under `.claude/agents/` are designed to work in any project. No hardcoded
paths, module names, or project-specific MCP tools. If an agent references a tool the
project doesn't have, it degrades gracefully.

### Notifications: WeChat + desktop + voice

`notify.sh` supports three channels for getting user attention:
- WeChat via Server酱 (key in `.claude/.env`, blocked from edits by pre-tool-use)
- macOS desktop notification via `osascript`
- Voice via `say` (two repetitions for audibility)

## Known Sharp Edges

- **`.env` is write-protected** by pre-tool-use.sh. To update credentials, edit it outside
  Claude Code or temporarily disable the hook.
- **force-push and direct main/master push are blocked** by pre-tool-use.sh. Use PR workflow.
- **Shell scripts must pass shellcheck** on every edit via post-tool-use.sh. False positives
  can be suppressed with `# shellcheck disable=SCXXXX` comments.
- **Agent files are loaded at session start.** After editing agent files on disk, restart
  the session or use `/agents` to reload.

## Notification Setup

| Channel | Method | Config |
|---------|--------|--------|
| WeChat | Server酱 | `SCT_SENDKEY` in `.claude/.env` |
| Desktop | macOS osascript | Built-in, no config |
| Voice | macOS `say` | Built-in, no config |
