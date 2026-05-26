# Git Conventions

## Commit messages

Use Conventional Commits format:

```
type(scope): description
```

Types: `feat`, `fix`, `docs`, `refactor`, `style`, `test`, `chore`, `perf`

Scopes map to project directories: `scripts`, `docker`, `ray`, `vllm`, `tools`, `docs`, `hooks`

## Commit rules

- Always create a NEW commit — never `--amend` unless explicitly asked
- Never skip pre-commit hooks (`--no-verify`)
- Never force-push to main/master
- Do not commit files that may contain secrets (`.env`, `credentials.json`, etc.)

## Pre-commit hooks

Do not modify the existing hook configuration in `.claude/settings.json` or `.claude/hooks/` unless the user explicitly requests it.
