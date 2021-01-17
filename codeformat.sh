#!/usr/bin/env bash

# we run clang-format and astyle twice to get stable format output

find src/ tools/ tests/ examples/ benchmark/ python/ -type f -name '*.c' -o -name '*.cpp' -o -name '*.cc' -o -name '*.h' | grep -v python/pybind11 | xargs -i clang-format -i {}
astyle -n -r "benchmark/*.h,*.cpp,*.cc" "src/*.h,*.cpp,*.cc" "tests/*.h,*.cpp,*.cc" "tools/*.h,*.cpp,*.cc" "examples/*.h,*.cpp,*.cc"
astyle -n -r "python/*.h,*.cpp,*.cc" --exclude=python/pybind11

find src/ tools/ tests/ examples/ benchmark/ python/ -type f -name '*.c' -o -name '*.cpp' -o -name '*.cc' -o -name '*.h' | grep -v python/pybind11 | xargs -i clang-format -i {}
astyle -n -r "benchmark/*.h,*.cpp,*.cc" "src/*.h,*.cpp,*.cc" "tests/*.h,*.cpp,*.cc" "tools/*.h,*.cpp,*.cc" "examples/*.h,*.cpp,*.cc"
astyle -n -r "python/*.h,*.cpp,*.cc" --exclude=python/pybind11
