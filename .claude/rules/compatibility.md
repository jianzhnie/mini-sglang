# Compatibility Constraints

Do NOT change these interfaces — they are depended upon by ops workflows.

## Must not change

- CLI parameters and env var names of all scripts
- `node_list.txt` parsing format: `awk 'NF && !/^#/ {print $1}'`
- `ssh_run` calling convention and `SSH_OPTS` word-split behavior in `common.sh`
- Device/driver mount paths in `ascend_infer_docker_run.sh` and `ascend_train_docker_run.sh`
- The `source` dependency chain: `common.sh` → `docker_env.sh` / `set_ray_env.sh` / `set_env.sh`

## Dependencies

- No new dependencies beyond: bash 4+, coreutils, openssh, docker, ray, vllm
- Must remain compatible with bash 4.2+
- Do not use bash 4.3+ features (e.g. `declare -A` for associative arrays, namerefs)

## Environment variables

- All scripts use `${VAR:-default}` pattern for config
- Override via env vars, env files, or CLI args
- When adding a new variable, always provide a sensible default
