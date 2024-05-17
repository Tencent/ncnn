# mlir2ncnn

## Compile

**Clone LLVM**
```bash
https://github.com/llvm/llvm-project.git
git checkout -b mlir <a_working_commit_id>
```
Current working commit id is 74e6030bcbcc8e628f9a99a424342a0c656456f9:
```bash
$ git log

commit 74e6030bcbcc8e628f9a99a424342a0c656456f9 (HEAD -> main, origin/main, origin/HEAD)
Author: Craig Topper <craig.topper@sifive.com>
Date:   Thu Mar 4 22:30:38 2021 -0800

    [TargetLowering] Use HandleSDNodes to prevent nodes from being deleted by recursive calls in getNegatedExpression.
```

It is determined by query lastest git commit date of `tools/mlir` directory.


**Compile mlir**
```bash
cd llvm-project
mkdir build
cd build
cmake -G Ninja -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="" -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_TESTS=OFF ../llvm/
ninja -j8
ninja install
```

**Compile mlir2ncnn**
```bash
cd tools/mlir
mkdir build
cd build
cmake .. -D LLVM_DIR=<path/to/your/llvm_install/lib/cmake/llvm>
make
```

## Usage

**Export `.mlir`**

See https://zhuanlan.zhihu.com/p/152535430


**Usage mlir2ncnn**

```bash
./mlir2ncnn pix2pix.mlir pix2pix.param pix2pix.bin
```
