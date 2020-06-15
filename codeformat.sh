#!/usr/bin/env bash

# we run clang-format and astyle twice to get stable format output

find src/ tools/ tests/ examples/ benchmark/ -type f -name '*.c' -o -name '*.cpp' -o -name '*.h' | xargs -i clang-format -i {}
astyle -n -r "benchmark/*.h,*.cpp" "src/*.h,*.cpp" "tests/*.h,*.cpp" "tools/*.h,*.cpp" "examples/*.h,*.cpp"

find src/ tools/ tests/ examples/ benchmark/ -type f -name '*.c' -o -name '*.cpp' -o -name '*.h' | xargs -i clang-format -i {}
astyle -n -r "benchmark/*.h,*.cpp" "src/*.h,*.cpp" "tests/*.h,*.cpp" "tools/*.h,*.cpp" "examples/*.h,*.cpp"
