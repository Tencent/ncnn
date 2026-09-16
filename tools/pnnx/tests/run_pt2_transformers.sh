#!/bin/bash
# Run the transformer-attention pt2 tests under a chosen python interpreter.
#
# These tests must be validated under BOTH major transformers lines before a
# push: transformers 5.x (the main dev venv) and transformers 4.x (the CI
# transformers4 job uses 4.48.3). the two versions emit different graphs (e.g.
# mt5/t5 serialize torch.arange under different overloads) and only one of them
# exposes a given loader gap, so a single-version run can stay green while the
# other version is broken.
#
# Usage (run from tools/pnnx/build/tests, like the other test runners):
#   PYTHON=/path/to/venv/bin/python ./run_pt2_transformers.sh
#   # default: PYTHON=python3
#
# Exit code 0 only when every test passes (like the CI loop).

if [ -z "$PYTHON" ]; then
    PYTHON=python3
fi

if [ ! -f ../src/pnnx ]; then
    echo "run this from tools/pnnx/build/tests (../src/pnnx not found)" >&2
    exit 2
fi
if ! ls ../../tests/test_transformers_*.py >/dev/null 2>&1; then
    echo "../../tests/test_transformers_*.py not found" >&2
    exit 2
fi

# the generated *_pnnx.py modules live in the cwd
export PYTHONPATH="."

total=0
pass=0
fail=0
for t in ../../tests/test_transformers_*.py; do
    total=$((total + 1))
    name=$(basename "$t")
    if out=$("$PYTHON" "$t" 2>&1); then
        pass=$((pass + 1))
        echo "PASS: $name"
    else
        rc=$?
        fail=$((fail + 1))
        echo "FAIL: $name (exit $rc)"
        echo "$out" | tail -40 | sed 's/^/    /'
    fi
done
echo "=== SUMMARY total=$total pass=$pass fail=$fail ($PYTHON) ==="
[ "$fail" -eq 0 ]
