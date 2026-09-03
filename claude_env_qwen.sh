# Claude Code environment for the local Qwen3.8 vLLM-Ascend service.
#
# Usage:
#   source ./claude_env_qwen.sh
#   claude --model qwen3.8
#
# The service exposes both the OpenAI-compatible and Anthropic Messages APIs
# on the same business port. Keep this URL free of a trailing /v1: Claude Code
# adds /v1/messages itself.

export ANTHROPIC_BASE_URL="${QWEN_ANTHROPIC_BASE_URL:-http://127.0.0.1:8022}"
export ANTHROPIC_API_KEY="${QWEN_ANTHROPIC_API_KEY:-dummy}"
export ANTHROPIC_AUTH_TOKEN="${QWEN_ANTHROPIC_AUTH_TOKEN:-dummy}"

# vLLM's --served-model-name in run_vllm.sh.
QWEN_MODEL="${QWEN_MODEL:-qwen3.8}"
export ANTHROPIC_MODEL="$QWEN_MODEL"
export ANTHROPIC_MODEL_SMALL="$QWEN_MODEL"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$QWEN_MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$QWEN_MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$QWEN_MODEL"
export CLAUDE_CODE_SUBAGENT_MODEL="$QWEN_MODEL"

# Tell Claude Code about the deployed 1M context window. This avoids the
# client's conservative 200K unknown-model limit while leaving the server as
# the source of truth for request validation.
export CLAUDE_CODE_MAX_CONTEXT_TOKENS="${QWEN_MAX_CONTEXT_TOKENS:-1000000}"
export CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1

# Qwen3.8's chat template accepts only xhigh, medium and low. Claude Code's
# high/max values are rejected by the template, so default to medium for an
# Agent-friendly throughput/quality balance. Set QWEN_REASONING_EFFORT=xhigh
# for the server's strongest reasoning mode, or low for lower latency.
QWEN_REASONING_EFFORT="${QWEN_REASONING_EFFORT:-medium}"
case "$QWEN_REASONING_EFFORT" in
    xhigh|medium|low)
        export CLAUDE_CODE_EFFORT_LEVEL="$QWEN_REASONING_EFFORT"
        ;;
    *)
        echo "ERROR: QWEN_REASONING_EFFORT must be xhigh, medium, or low (got $QWEN_REASONING_EFFORT)" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

echo "[claude-qwen] endpoint=${ANTHROPIC_BASE_URL} model=${QWEN_MODEL} effort=${CLAUDE_CODE_EFFORT_LEVEL}"
