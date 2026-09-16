#!/bin/bash
# Pre-push dual-line transformers regression.
#
# The transformer-attention pt2 tests must pass under BOTH supported
# transformers major lines before pushing:
#   - transformers 5.x   (main dev venv: torch 2.13, transformers 5.16.x)
#   - transformers 4.48.3 (the CI transformers4 job)
#
# The two versions emit different graphs (e.g. mt5/t5 serialize torch.arange
# under different overloads) and a loader gap can surface on only one line, so
# a single green line is not enough.
#
# Usage (from anywhere in the repo):
#   tools/pnnx/tests/run_pt2_transformers_dual.sh
# Override the interpreter paths with PNNX_PT2_VENV5 / PNNX_PT2_VENV4.
# Exit code 0 only when every test passes on both lines.
#
# To wire it as a git pre-push hook:
#   ln -sf ../../../../tools/pnnx/tests/run_pt2_transformers_dual.sh .git/hooks/pre-push

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT/tools/pnnx/build/tests" || {
    echo "tools/pnnx/build/tests not found - build pnnx first" >&2
    exit 2
}

VENV5="${PNNX_PT2_VENV5:-/home/edwards/tx_opensource/venv}"
VENV4="${PNNX_PT2_VENV4:-/home/edwards/tx_opensource/venv_tf4483}"

overall=0
for v in "$VENV5" "$VENV4"; do
    if [ ! -x "$v/bin/python" ]; then
        echo "missing venv python: $v/bin/python (set PNNX_PT2_VENV5 / PNNX_PT2_VENV4)" >&2
        exit 2
    fi
    tfver=$("$v/bin/python" -c 'import transformers; print(transformers.__version__)' 2>/dev/null)
    echo "########## transformers $tfver ($v) ##########"
    PYTHON="$v/bin/python" "$REPO_ROOT/tools/pnnx/tests/run_pt2_transformers.sh"
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "########## transformers $tfver: PASS ##########"
    else
        echo "########## transformers $tfver: FAIL ##########"
        overall=1
    fi
done

[ "$overall" -eq 0 ]
