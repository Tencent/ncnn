mkdir build-vs2019
cd build-vs2019
cmake -G "Visual Studio 16 2019" -A x64 -DNCNN_PIXEL_DRAWING=OFF -DNCNN_PIXEL_ROTATE=OFF -DNCNN_PIXEL_AFFINE=OFF -DNCNN_ARM82=OFF  -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_EXAMPLES=OFF ..
cmake --build . --config Release
cmake --install . --config Release
cd ..

cp xmake/packages ~/.KnightRepo -rf
mkdir -p ~/.KnightRepo/packages/n/ncnn/windows/x64/release/
cp build-vs2019/install/* ~/.KnightRepo/packages/n/ncnn/windows/x64/release/ -rf