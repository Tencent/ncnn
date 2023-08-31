#!/usr/bin/env bash

# we run clang-format and astyle twice to get stable format output

format_code() {
    find src/ tools/ tests/ examples/ benchmark/ python/ -type f -name '*.c' -o -name '*.cpp' -o -name '*.cc' -o -name '*.h' | grep -v python/pybind11 | grep -v stb_image | xargs -i clang-format -i {}
    astyle -n -r "benchmark/*.h,*.cpp,*.cc" "tests/*.h,*.cpp,*.cc" "tools/*.h,*.cpp,*.cc" "examples/*.h,*.cpp,*.cc"
    astyle -n -r "src/*.h,*.cpp,*.cc" --exclude=src/stb_image.h --exclude=src/stb_image_write.h
    astyle -n -r "python/*.h,*.cpp,*.cc" --exclude=python/pybind11
}

format_code || { echo 'Formatting failed' ; exit 1; } #first time execute
format_code || { echo 'Formatting failed' ; exit 1; } #second time execute
