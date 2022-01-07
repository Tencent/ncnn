emsdk_cmake="D:\\hqj\\test\\Pubilc\\emsdk\\upstream\\emscripten\\cmake\\Modules\\Platform\\Emscripten.cmake"
cp xmake/packages ~/.KnightRepo -rf

cpinstallfile() {
    if [ -d ~/.KnightRepo/packages/n/$1/wasm/wasm32/release ] 
    then
        rm ~/.KnightRepo/packages/n/$1/wasm/wasm32/release/* -rf
    else 
        mkdir -p ~/.KnightRepo/packages/n/$1/wasm/wasm32/release/
    fi
    cp install/* ~/.KnightRepo/packages/n/$1/wasm/wasm32/release/ -rf
}

echo "Build default wasm..."
mkdir -p build-wasm
cd build-wasm
cmake -G "MinGW Makefiles" -DCMAKE_TOOLCHAIN_FILE=$emsdk_cmake -DNCNN_THREADS=OFF -DNCNN_OPENMP=OFF -DNCNN_SIMPLEOMP=OFF -DNCNN_RUNTIME_CPU=OFF -DNCNN_SSE2=OFF -DNCNN_AVX2=OFF -DNCNN_AVX=OFF -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF ..
cmake --build . -j 4
cmake --build . --target install

cpinstallfile ncnn
cd ..


echo "Build wasm with threads..."
mkdir -p build-wasm-threads
cd build-wasm-threads
cmake -G "MinGW Makefiles" -DCMAKE_TOOLCHAIN_FILE=$emsdk_cmake -DNCNN_THREADS=ON -DNCNN_OPENMP=ON -DNCNN_SIMPLEOMP=ON -DNCNN_RUNTIME_CPU=OFF -DNCNN_SSE2=OFF -DNCNN_AVX2=OFF -DNCNN_AVX=OFF -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF ..
cmake --build . -j 4
cmake --build . --target install

cpinstallfile ncnn-wasm-threads
cd ..


echo "Build wasm with simd..."
mkdir -p build-wasm-simd
cd build-wasm-simd
cmake -G "MinGW Makefiles" -DCMAKE_TOOLCHAIN_FILE=$emsdk_cmake -DNCNN_THREADS=OFF -DNCNN_OPENMP=OFF -DNCNN_SIMPLEOMP=OFF -DNCNN_RUNTIME_CPU=OFF -DNCNN_SSE2=ON -DNCNN_AVX2=OFF -DNCNN_AVX=OFF     -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF ..
cmake --build . -j 4
cmake --build . --target install

cpinstallfile ncnn-wasm-simd
cd ..

echo "Build wasm with simd and threads..."
mkdir -p build-wasm-simd
cd build-wasm-simd
cmake -G "MinGW Makefiles" -DCMAKE_TOOLCHAIN_FILE=$emsdk_cmake -DNCNN_THREADS=ON -DNCNN_OPENMP=ON -DNCNN_SIMPLEOMP=ON -DNCNN_RUNTIME_CPU=OFF -DNCNN_SSE2=ON -DNCNN_AVX2=OFF -DNCNN_AVX=OFF -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF ..
cmake --build . -j 4
cmake --build . --target install

cpinstallfile ncnn-wasm-simd-threads
cd ..
