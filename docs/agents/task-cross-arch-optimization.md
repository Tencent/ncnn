# Task: Add Optimization for Cross-Compiled Architectures (ARM, RISC-V, etc.)

For architectures that require cross-compilation and QEMU simulation for testing:

1. **Create the arch-specific files:**
   - ARM: `src/layer/arm/<name>_arm.h` and `src/layer/arm/<name>_arm.cpp`
   - RISC-V: `src/layer/riscv/<name>_riscv.h` and `src/layer/riscv/<name>_riscv.cpp`
   - LoongArch: `src/layer/loongarch/<name>_loongarch.h` and `src/layer/loongarch/<name>_loongarch.cpp`
   - MIPS: `src/layer/mips/<name>_mips.h` and `src/layer/mips/<name>_mips.cpp`

2. **ARM NEON example pattern:**
   ```cpp
   #include "newop_arm.h"
   #if __ARM_NEON
   #include <arm_neon.h>
   #endif

   namespace ncnn {
   NewOp_arm::NewOp_arm()
   {
   #if __ARM_NEON
       support_packing = true;
   #if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
       support_fp16_storage = true;
   #endif
   #if NCNN_BF16
       support_bf16_storage = true;
   #endif
   #endif // __ARM_NEON
   }

   int NewOp_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
   {
   #if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
       if (opt.use_fp16_storage)
           return forward_fp16s(bottom_blob, top_blob, opt);
   #endif
       // NEON implementation with float32x4_t
       return 0;
   }
   } // namespace ncnn
   ```

   For sub-ISA dispatch (e.g., `asimdhp`, `asimddp`, `bf16`, `i8mm`, `sve`), create separate source files like `<name>_arm_asimdhp.cpp`. The build system compiles them with appropriate `-march=` flags.

3. **RISC-V Vector example pattern:**
   ```cpp
   #include "newop_riscv.h"
   #if __riscv_vector
   #include <riscv_vector.h>
   #endif

   namespace ncnn {
   NewOp_riscv::NewOp_riscv()
   {
   #if __riscv_vector
       support_packing = true;
   #if __riscv_zvfh
       support_fp16_storage = true;
   #endif
   #endif
   }

   int NewOp_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
   {
       const int packn = csrr_vlenb() / 4;  // Elements per vector register
       const size_t vl = __riscv_vsetvl_e32m1(packn);
       // Use RVV intrinsics
       return 0;
   }
   } // namespace ncnn
   ```

4. **Build and test:**
   ```bash
   # ARM AArch64
   cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake \
         -DNCNN_BUILD_TESTS=ON ..
   cmake --build . -j$(nproc)
   TESTS_EXECUTABLE_LOADER=qemu-aarch64-static \
   TESTS_EXECUTABLE_LOADER_ARGUMENTS="-L;/usr/aarch64-linux-gnu" \
   ctest --output-on-failure -j8

   # RISC-V 64 with RVV
   export RISCV_ROOT_PATH=/path/to/riscv
   cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/riscv64-unknown-linux-gnu.toolchain.cmake \
         -DNCNN_RVV=ON -DNCNN_ZFH=ON -DNCNN_ZVFH=ON -DNCNN_BUILD_TESTS=ON ..
   cmake --build . -j$(nproc)
   TESTS_EXECUTABLE_LOADER=qemu-riscv64 \
   TESTS_EXECUTABLE_LOADER_ARGUMENTS="-cpu;rv64,v=true,zfh=true,zvfh=true,vlen=256,elen=64,vext_spec=v1.0;-L;$RISCV_ROOT_PATH/sysroot" \
   ctest --output-on-failure -j8
   ```

5. **No CMakeLists.txt changes needed** for the layer itself — the build system auto-detects arch-specific source files.

## Cross-Compilation Tips

- Toolchain files are in `toolchains/` — study them to understand available targets
- `NCNN_RUNTIME_CPU=ON` enables multi-ISA dispatch within one binary
- CI tests many arch/ISA combinations — check `.github/workflows/test-coverage.yml` for the full matrix
- QEMU user-mode emulation is used for testing cross-compiled binaries
- Intel SDE is used for testing x86 ISA extensions not available on the CI host
