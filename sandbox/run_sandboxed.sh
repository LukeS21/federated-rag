#!/bin/bash
# Federated RAG — Native macOS Sandbox Launcher (Phase 5)
#
# Runs the Federated RAG pipeline under macOS Seatbelt sandboxing,
# providing kernel-level filesystem and network isolation without Docker.
#
# Usage:
#   ./sandbox/run_sandboxed.sh              # Survey Mode demo (default)
#   ./sandbox/run_sandboxed.sh deep         # Deep Mode demo
#   ./sandbox/run_sandboxed.sh test         # Run tests
#   ./sandbox/run_sandboxed.sh verify       # Run Phase 5 verification
#   ./sandbox/run_sandboxed.sh --no-sandbox # Run without sandbox (dev mode)
#
# Prerequisites:
#   - macOS 10.12+ (Sierra or later for Seatbelt)
#   - Python 3.12 with dependencies installed
#   - Ollama running locally with models pulled

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SB_PROFILE="$SCRIPT_DIR/federated_rag.sb"
USE_SANDBOX=true
MODE="${1:-survey}"

# Parse flags
if [[ "${1:-}" == "--no-sandbox" ]]; then
    USE_SANDBOX=false
    MODE="${2:-survey}"
fi

# ── Environment ──────────────────────────────────────────────────────────
export LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
export OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
export PROJECT_DIR="projects/default"
export SECURITY_AUDIT_LOG="${PROJECT_ROOT}/logs/security_audit.log"
export BOUNDARY_SCRUB_PATTERNS="${PROJECT_ROOT}/config/scrub_patterns.txt"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"

# ── Verify prerequisites ─────────────────────────────────────────────────
if [ "$USE_SANDBOX" = true ]; then
    if ! command -v sandbox-exec &>/dev/null; then
        echo "ERROR: sandbox-exec not found. This script requires macOS."
        echo "Run with --no-sandbox to skip sandboxing:"
        echo "  $0 --no-sandbox $MODE"
        exit 1
    fi

    if [ ! -f "$SB_PROFILE" ]; then
        echo "ERROR: Sandbox profile not found at $SB_PROFILE"
        exit 1
    fi

    echo "=== Federated RAG Sandboxed Runner ==="
    echo "Sandbox profile: $SB_PROFILE"
    echo "Mode:            $MODE"
    echo "LLM Provider:    $LLM_PROVIDER"
    echo ""
    echo "The app is restricted to:"
    echo "  - Read/write: $PROJECT_ROOT/projects, $PROJECT_ROOT/logs"
    echo "  - Read-only:  $PROJECT_ROOT/src, $PROJECT_ROOT/papers, $PROJECT_ROOT/config"
    echo "  - Network:    localhost only (Ollama on 127.0.0.1:11434)"
    echo ""
else
    echo "=== Federated RAG (unsandboxed dev mode) ==="
fi

# ── Run ──────────────────────────────────────────────────────────────────
case "$MODE" in
    survey)
        CMD=(python "$PROJECT_ROOT/phase4_demo.py")
        ;;
    deep)
        CMD=(python "$PROJECT_ROOT/phase3_demo.py")
        ;;
    test)
        CMD=(python -m pytest "$PROJECT_ROOT/tests/" -v)
        ;;
    verify)
        CMD=(python "$PROJECT_ROOT/phase5_verify.py")
        ;;
    benchmark)
        CMD=(python "$PROJECT_ROOT/phase4_benchmark.py")
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Valid modes: survey, deep, test, verify, benchmark"
        exit 1
        ;;
esac

if [ "$USE_SANDBOX" = true ]; then
    exec sandbox-exec -f "$SB_PROFILE" "${CMD[@]}"
else
    exec "${CMD[@]}"
fi
