# Task: Add x86 SIMD Optimization for an Existing Operator

1. **Create the arch-specific header** — `src/layer/x86/<name>_x86.h`
   ```cpp
   #ifndef LAYER_NEWOP_X86_H
   #define LAYER_NEWOP_X86_H
   #include "newop.h"
   namespace ncnn {
   class NewOp_x86 : public NewOp
   {
   public:
       NewOp_x86();
       virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
   };
   } // namespace ncnn
   #endif
   ```

2. **Implement with SIMD intrinsics** — `src/layer/x86/<name>_x86.cpp`
   ```cpp
   #include "newop_x86.h"
   #if __SSE2__
   #include <emmintrin.h>
   #if __AVX__
   #include <immintrin.h>
   #endif
   #endif

   namespace ncnn {
   NewOp_x86::NewOp_x86()
   {
   #if __SSE2__
       support_packing = true;  // Accept packed Mat
   #endif
   }

   int NewOp_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
   {
       int elempack = bottom_blob.elempack;
       int size = w * h * d * elempack;

       // Process with widest available SIMD first, then fall through
       int i = 0;
   #if __SSE2__
   #if __AVX__
   #if __AVX512F__
       for (; i + 15 < size; i += 16)
       {
           __m512 _v = _mm512_loadu_ps(ptr + i);
           // ... AVX-512 processing
           _mm512_storeu_ps(outptr + i, _v);
       }
   #endif // __AVX512F__
       for (; i + 7 < size; i += 8)
       {
           __m256 _v = _mm256_loadu_ps(ptr + i);
           // ... AVX processing
           _mm256_storeu_ps(outptr + i, _v);
       }
   #endif // __AVX__
       for (; i + 3 < size; i += 4)
       {
           __m128 _v = _mm_loadu_ps(ptr + i);
           // ... SSE processing
           _mm_storeu_ps(outptr + i, _v);
       }
   #endif // __SSE2__
       for (; i < size; i++)
       {
           // scalar fallback
       }
       return 0;
   }
   } // namespace ncnn
   ```

3. **No CMakeLists.txt changes needed** — the `ncnn_add_layer()` macro automatically detects `src/layer/x86/<name>_x86.cpp` and compiles it.

4. **Run tests** — the existing `test_<name>` will automatically test the x86 path on x86 machines.

## SIMD Code Conventions

- Use nested `#if` guards: `#if __SSE2__` → `#if __AVX__` → `#if __AVX512F__`
- Process widest SIMD first, fall through to narrower, end with scalar
- Set `support_packing = true` in constructor to accept packed data
- Account for `elempack` in size calculations: `int size = w * h * d * elempack`
- For runtime CPU dispatch (`NCNN_RUNTIME_CPU=ON`), the build system auto-generates variant files with appropriate compiler flags
