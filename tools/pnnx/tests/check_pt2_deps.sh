#!/bin/bash
# pnnx pt2 parsing path zero-third-party-dependency check
# verifies new C++ files only include pnnx internal headers and the C++ stdlib
# derive the pnnx source dir from this script's location so the check works on
# any checkout (not just the author's machine)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC="$SCRIPT_DIR/../src"

files=(
    "$SRC/load_exportedprogram.cpp"
    "$SRC/load_exportedprogram.h"
    "$SRC/load_exportedprogram_legacy.cpp"
    "$SRC/load_exportedprogram_legacy.h"
    "$SRC/load_pt2_defaults.cpp"
    "$SRC/load_pt2_defaults.h"
    "$SRC/load_pt2_serde.cpp"
    "$SRC/load_pt2_serde.h"
    "$SRC/load_pt2_tensor.cpp"
    "$SRC/load_pt2_tensor.h"
    "$SRC/pnnx_json.cpp"
    "$SRC/pnnx_json.h"
    "$SRC/pass_level2/eliminate_noop_copy.cpp"
    "$SRC/pass_level2/eliminate_noop_copy.h"
)

# pnnx internal headers (quoted form)
internal='load_exportedprogram\.h|load_exportedprogram_legacy\.h|load_pt2_defaults\.h|load_pt2_serde\.h|load_pt2_tensor\.h|pnnx_json\.h|storezip\.h|ir\.h|pass_level2\.h|eliminate_noop_copy\.h'

# C++ standard library headers (angle-bracket form)
stdlib='cstdio|stdio\.h|string\.h|stdlib\.h|math\.h|stdint\.h|stddef\.h|string|vector|map|algorithm|cmath|stddef|climits|limits|complex'

violations=0
for f in "${files[@]}"; do
    while IFS= read -r line; do
        # extract the include target
        inc=$(echo "$line" | sed -nE 's/^#include [<"]([^>"]+)[>"]/\1/p')
        [ -z "$inc" ] && continue
        if echo "$line" | grep -q '"'; then
            # quoted form: must be a pnnx internal header
            if ! echo "$inc" | grep -qE "^($internal)$"; then
                echo "VIOLATION $f : non-internal header $inc"
                violations=$((violations+1))
            fi
        else
            # angle-bracket form: must be standard library
            if ! echo "$inc" | grep -qE "^($stdlib)$"; then
                echo "VIOLATION $f : non-stdlib header $inc"
                violations=$((violations+1))
            fi
        fi
    done < "$f"
done

if [ "$violations" -eq 0 ]; then
    echo "PASS: pt2 parsing path has zero third-party dependencies (all ${#files[@]} files passed)"
else
    echo "FAIL: $violations non-standard dependency(ies)"
    exit 1
fi
