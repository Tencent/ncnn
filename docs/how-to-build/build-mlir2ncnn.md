# mlir2ncnn

## Compile

**Clone LLVM**
```bash
https://github.com/llvm/llvm-project.git
git checkout -b mlir <a_working_commit_id>
```
Current working commit id is 7c15e0f64ccc79a53ed2db258f1cb58ec452a957:
```
$ git log

commit 7c15e0f64ccc79a53ed2db258f1cb58ec452a957 (HEAD -> 01-26)
Author: MaheshRavishankar <ravishankarm@google.com>
Date:   Tue Jan 26 23:21:33 2021 -0800

    [mlir][Linalg] Add canonicalization for init_tensor -> subtensor op.
    
    Differential Revision: https://reviews.llvm.org/D95305
```

It is determined by query lastest git commit date of `tools/mlir` directory.


**Compile mlir**
```bash
cd llvm-project
mkdir build
cd build
cmake -G Ninja -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="" -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_TESTS=OFF ../llvm/
make -j8
make install
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

```
./mlir2ncnn pix2pix.mlir pix2pix.param pix2pix.bin
```
