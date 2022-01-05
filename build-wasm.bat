mkdir build-wasm
cd build-wasm
cmake -G "MinGW Makefiles" -DCMAKE_TOOLCHAIN_FILE=D:\hqj\test\Pubilc\emsdk\upstream\emscripten\cmake\Modules\Platform\Emscripten.cmake -DNCNN_THREADS=OFF -DNCNN_OPENMP=OFF -DNCNN_SIMPLEOMP=OFF -DNCNN_RUNTIME_CPU=OFF -DNCNN_SSE2=OFF -DNCNN_AVX2=OFF -DNCNN_AVX=OFF -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF ..
cmake --build . -j 4
cmake --build . --target install