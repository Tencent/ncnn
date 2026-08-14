#ifndef SDPA_X86_INT8_H
#define SDPA_X86_INT8_H

static inline signed char float2int8(float v)
{
    int int32 = static_cast<int>(roundf(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX__ && !__AVX512VNNI__
void decode_qk_dot_int8_avx512vnni(float* s, const signed char* q, const signed char* K, const float* qscales, const float* kscales, int n_start, int block_n, int d, float scale);
void qk_int8_gemm_row_avx512vnni(float* s_row, const signed char* q_row, const signed char* K, float qscale, const float* kscales, int n, int d, float scale);
void qk_int8_gemm_tiled_avx512vnni(float* S, const signed char* Q, const signed char* K, const float* qscales, const float* kscales, int m, int n, int d, float scale);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
void decode_qk_dot_int8_avxvnniint8(float* s, const signed char* q, const signed char* K, const float* qscales, const float* kscales, int n_start, int block_n, int d, float scale);
void qk_int8_gemm_row_avxvnniint8(float* s_row, const signed char* q_row, const signed char* K, float qscale, const float* kscales, int n, int d, float scale);
void qk_int8_gemm_tiled_avxvnniint8(float* S, const signed char* Q, const signed char* K, const float* qscales, const float* kscales, int m, int n, int d, float scale);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
void decode_qk_dot_int8_avxvnni(float* s, const signed char* q, const signed char* K, const float* qscales, const float* kscales, int n_start, int block_n, int d, float scale);
void qk_int8_gemm_row_avxvnni(float* s_row, const signed char* q_row, const signed char* K, float qscale, const float* kscales, int n, int d, float scale);
void qk_int8_gemm_tiled_avxvnni(float* S, const signed char* Q, const signed char* K, const float* qscales, const float* kscales, int m, int n, int d, float scale);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__
void dynamic_quantize_blockwise_avx2(const float* src, signed char* dst, float* scales, int width);
void dynamic_quantize_rowwise_avx2(const float* src, signed char* dst, float* scale, int width);
int qk_int8_dot_block_avx2(const signed char* a, const signed char* b, int len);
void decode_qk_dot_int8_avx2(float* s, const signed char* q, const signed char* K, const float* qscales, const float* kscales, int n_start, int block_n, int d, float scale);
void qk_int8_gemm_row_avx2(float* s_row, const signed char* q_row, const signed char* K, float qscale, const float* kscales, int n, int d, float scale);
void qk_int8_gemm_tiled_avx2(float* S, const signed char* Q, const signed char* K, const float* qscales, const float* kscales, int m, int n, int d, float scale);
void decode_pv_gemv_int8_avx2(float* out, const float* s, const signed char* V, const float* vscales, int n_start, int block_n, int out_d);
void pv_float_int8_gemm_row_avx2(float* out, const float* p_row, const signed char* V, const float* vscales, int n, int out_d);
void pv_float_int8_fma_block_avx2(float* out, float p_invscale, const signed char* v, int len);
void pv_float_int8_gemm_tile_avx2(float* O, const float* P, const signed char* V, const float* vscales, int block_m, int block_n, int out_embed_dim);
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
void decode_qk_dot_int8_xop(float* s, const signed char* q, const signed char* K, const float* qscales, const float* kscales, int n_start, int block_n, int d, float scale);
void qk_int8_gemm_row_xop(float* s_row, const signed char* q_row, const signed char* K, float qscale, const float* kscales, int n, int d, float scale);
void qk_int8_gemm_tiled_xop(float* S, const signed char* Q, const signed char* K, const float* qscales, const float* kscales, int m, int n, int d, float scale);
#endif

static void dynamic_quantize_blockwise_scalar_kernel(const float* src, signed char* dst, float* scales, int width)
{
    const int block_size = 32;
    int num_blocks = (width + block_size - 1) / block_size;
    for (int b = 0; b < num_blocks; b++)
    {
        int start = b * block_size;
        int end = start + block_size < width ? start + block_size : width;
        float absmax = 0.f;
        for (int i = start; i < end; i++)
        {
            absmax = std::max(absmax, fabsf(src[i]));
        }
        float scale = absmax == 0.f ? 1.f : 127.f / absmax;
        scales[b] = scale;
        for (int i = start; i < end; i++)
        {
            dst[i] = float2int8(src[i] * scale);
        }
    }
}

#if __SSE2__
static inline void dynamic_quantize_blockwise_sse2_kernel(const float* src, signed char* dst, float* scales, int width)
{
    const int block_size = 32;
    int num_blocks = (width + block_size - 1) / block_size;
    __m128 sign_mask = _mm_castsi128_ps(_mm_set1_epi32(1 << 31));
    for (int b = 0; b < num_blocks; b++)
    {
        int start = b * block_size;
        int end = start + block_size < width ? start + block_size : width;
        float absmax = 0.f;
        int i = start;
        __m128 vmax = _mm_setzero_ps();
        for (; i + 3 < end; i += 4)
        {
            __m128 x = _mm_loadu_ps(src + i);
            __m128 ax = _mm_andnot_ps(sign_mask, x);
            vmax = _mm_max_ps(vmax, ax);
        }
        absmax = _mm_reduce_max_ps(vmax);
        for (; i < end; i++)
        {
            absmax = std::max(absmax, fabsf(src[i]));
        }
        float scale = absmax == 0.f ? 1.f : 127.f / absmax;
        scales[b] = scale;
        __m128 vscale = _mm_set1_ps(scale);
        i = start;
        for (; i + 15 < end; i += 16)
        {
            __m128 x0 = _mm_loadu_ps(src + i);
            __m128 x1 = _mm_loadu_ps(src + i + 4);
            __m128 x2 = _mm_loadu_ps(src + i + 8);
            __m128 x3 = _mm_loadu_ps(src + i + 12);
            x0 = _mm_mul_ps(x0, vscale);
            x1 = _mm_mul_ps(x1, vscale);
            x2 = _mm_mul_ps(x2, vscale);
            x3 = _mm_mul_ps(x3, vscale);
            __m128i v8 = float2int8_sse(x0, x1, x2, x3);
            _mm_storeu_si128((__m128i*)(dst + i), v8);
        }
        for (; i + 3 < end; i += 4)
        {
            __m128 x = _mm_loadu_ps(src + i);
            x = _mm_mul_ps(x, vscale);
            int32_t v4 = float2int8_sse(x);
            *(int32_t*)(dst + i) = v4;
        }
        for (; i < end; i++)
        {
            dst[i] = float2int8(src[i] * scale);
        }
    }
}
#endif // __SSE2__

#if __AVX2__
inline void dynamic_quantize_blockwise_avx2_kernel(const float* src, signed char* dst, float* scales, int width)
{
    const int block_size = 32;
    int num_blocks = (width + block_size - 1) / block_size;
    __m256 sign_mask = _mm256_set1_ps(-0.f);
    for (int b = 0; b < num_blocks; b++)
    {
        int start = b * block_size;
        int end = start + block_size < width ? start + block_size : width;
        float absmax = 0.f;
        int i = start;
        __m256 vmax = _mm256_setzero_ps();
        for (; i + 7 < end; i += 8)
        {
            __m256 x = _mm256_loadu_ps(src + i);
            __m256 ax = _mm256_andnot_ps(sign_mask, x);
            vmax = _mm256_max_ps(vmax, ax);
        }
        absmax = _mm256_reduce_max_ps(vmax);
        for (; i < end; i++)
        {
            absmax = std::max(absmax, fabsf(src[i]));
        }
        float scale = absmax == 0.f ? 1.f : 127.f / absmax;
        scales[b] = scale;
        __m256 vscale = _mm256_set1_ps(scale);
        i = start;
        for (; i + 31 < end; i += 32)
        {
            __m256 x0 = _mm256_loadu_ps(src + i);
            __m256 x1 = _mm256_loadu_ps(src + i + 8);
            __m256 x2 = _mm256_loadu_ps(src + i + 16);
            __m256 x3 = _mm256_loadu_ps(src + i + 24);
            __m256i y0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(x0, vscale), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i y1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(x1, vscale), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i y2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(x2, vscale), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i y3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(x3, vscale), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m128i a0 = _mm256_castsi256_si128(y0);
            __m128i a1 = _mm256_extractf128_si256(y0, 1);
            __m128i a2 = _mm256_castsi256_si128(y1);
            __m128i a3 = _mm256_extractf128_si256(y1, 1);
            __m128i a4 = _mm256_castsi256_si128(y2);
            __m128i a5 = _mm256_extractf128_si256(y2, 1);
            __m128i a6 = _mm256_castsi256_si128(y3);
            __m128i a7 = _mm256_extractf128_si256(y3, 1);
            __m128i b0 = _mm_packs_epi32(a0, a1);
            __m128i b1 = _mm_packs_epi32(a2, a3);
            __m128i b2 = _mm_packs_epi32(a4, a5);
            __m128i b3 = _mm_packs_epi32(a6, a7);
            __m128i c0 = _mm_packs_epi16(b0, b1);
            __m128i c1 = _mm_packs_epi16(b2, b3);
            _mm_storeu_si128((__m128i*)(dst + i), c0);
            _mm_storeu_si128((__m128i*)(dst + i + 16), c1);
        }
        for (; i + 7 < end; i += 8)
        {
            __m256 x = _mm256_loadu_ps(src + i);
            __m256i y = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(x, vscale), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m128i a0 = _mm256_castsi256_si128(y);
            __m128i a1 = _mm256_extractf128_si256(y, 1);
            __m128i b0 = _mm_packs_epi32(a0, a1);
            __m128i c0 = _mm_packs_epi16(b0, b0);
            int64_t v8 = _mm_cvtsi128_si64(c0);
            *(int64_t*)(dst + i) = v8;
        }
        for (; i < end; i++)
        {
            dst[i] = float2int8(src[i] * scale);
        }
    }
}
#endif // __AVX2__

#if __AVX512F__
static inline void dynamic_quantize_blockwise_avx512_kernel(const float* src, signed char* dst, float* scales, int width)
{
    const int block_size = 32;
    int num_blocks = (width + block_size - 1) / block_size;
    __m512 sign_mask = _mm512_set1_ps(-0.f);
    for (int b = 0; b < num_blocks; b++)
    {
        int start = b * block_size;
        int end = start + block_size < width ? start + block_size : width;
        float absmax = 0.f;
        int i = start;
        __m512 vmax = _mm512_setzero_ps();
        for (; i + 15 < end; i += 16)
        {
            __m512 x = _mm512_loadu_ps(src + i);
            __m512 ax = _mm512_andnot_ps(sign_mask, x);
            vmax = _mm512_max_ps(vmax, ax);
        }
        absmax = _mm512_comp_reduce_max_ps(vmax);
        for (; i < end; i++)
        {
            absmax = std::max(absmax, fabsf(src[i]));
        }
        float scale = absmax == 0.f ? 1.f : 127.f / absmax;
        scales[b] = scale;
        __m512 vscale = _mm512_set1_ps(scale);
        i = start;
        for (; i + 15 < end; i += 16)
        {
            __m512 x = _mm512_loadu_ps(src + i);
            __m512i y = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(x, vscale), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m128i a0 = _mm512_extracti32x4_epi32(y, 0);
            __m128i a1 = _mm512_extracti32x4_epi32(y, 1);
            __m128i a2 = _mm512_extracti32x4_epi32(y, 2);
            __m128i a3 = _mm512_extracti32x4_epi32(y, 3);
            __m128i b0 = _mm_packs_epi32(a0, a1);
            __m128i b1 = _mm_packs_epi32(a2, a3);
            __m128i c0 = _mm_packs_epi16(b0, b1);
            _mm_storeu_si128((__m128i*)(dst + i), c0);
        }
        for (; i < end; i++)
        {
            dst[i] = float2int8(src[i] * scale);
        }
    }
}
#endif // __AVX512F__

static void dynamic_quantize_blockwise(const float* src, signed char* dst, float* scales, int width)
{
#if __AVX512F__
    dynamic_quantize_blockwise_avx512_kernel(src, dst, scales, width);
#else
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        dynamic_quantize_blockwise_avx2(src, dst, scales, width);
        return;
    }
#endif
#if __AVX2__
    dynamic_quantize_blockwise_avx2(src, dst, scales, width);
#elif __SSE2__
    dynamic_quantize_blockwise_sse2_kernel(src, dst, scales, width);
#else
    dynamic_quantize_blockwise_scalar_kernel(src, dst, scales, width);
#endif
#endif
}

static inline void dynamic_quantize_rowwise_scalar_kernel(const float* src, signed char* dst, float* scale, int width)
{
    float absmax = 0.f;
    for (int i = 0; i < width; i++)
    {
        absmax = std::max(absmax, fabsf(src[i]));
    }
    float s = absmax == 0.f ? 1.f : 127.f / absmax;
    *scale = s;
    for (int i = 0; i < width; i++)
    {
        dst[i] = float2int8(src[i] * s);
    }
}

#if __SSE2__
static inline void dynamic_quantize_rowwise_sse2_kernel(const float* src, signed char* dst, float* scale, int width)
{
    __m128 sign_mask = _mm_set1_ps(-0.f);
    __m128 vmax = _mm_setzero_ps();
    int i = 0;
    for (; i + 3 < width; i += 4)
    {
        __m128 x = _mm_loadu_ps(src + i);
        vmax = _mm_max_ps(vmax, _mm_andnot_ps(sign_mask, x));
    }
    float absmax = _mm_reduce_max_ps(vmax);
    for (; i < width; i++)
    {
        absmax = std::max(absmax, fabsf(src[i]));
    }
    float s = absmax == 0.f ? 1.f : 127.f / absmax;
    *scale = s;
    __m128 vscale = _mm_set1_ps(s);
    i = 0;
    for (; i + 3 < width; i += 4)
    {
        __m128 x = _mm_loadu_ps(src + i);
        __m128i yi = _mm_cvtps_epi32(_mm_mul_ps(x, vscale));
        (void)yi;
        dst[i + 0] = float2int8(src[i + 0] * s);
        dst[i + 1] = float2int8(src[i + 1] * s);
        dst[i + 2] = float2int8(src[i + 2] * s);
        dst[i + 3] = float2int8(src[i + 3] * s);
    }
    for (; i < width; i++)
    {
        dst[i] = float2int8(src[i] * s);
    }
}
#endif

#if __AVX2__
inline void dynamic_quantize_rowwise_avx2_kernel(const float* src, signed char* dst, float* scale, int width)
{
    __m256 sign_mask = _mm256_set1_ps(-0.f);
    __m256 vmax = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < width; i += 8)
    {
        __m256 x = _mm256_loadu_ps(src + i);
        vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign_mask, x));
    }
    float absmax = _mm256_reduce_max_ps(vmax);
    for (; i < width; i++)
    {
        absmax = std::max(absmax, fabsf(src[i]));
    }
    float s = absmax == 0.f ? 1.f : 127.f / absmax;
    *scale = s;
    __m256 vscale = _mm256_set1_ps(s);
    i = 0;
    for (; i + 7 < width; i += 8)
    {
        __m256 x = _mm256_loadu_ps(src + i);
        __m256i y = _mm256_cvtps_epi32(_mm256_mul_ps(x, vscale));
        __m128i a0 = _mm256_extracti128_si256(y, 0);
        __m128i a1 = _mm256_extracti128_si256(y, 1);
        __m128i b0 = _mm_packs_epi32(a0, a1);
        _mm_storel_epi64((__m128i*)(dst + i), _mm_packs_epi16(b0, b0));
    }
    for (; i < width; i++)
    {
        dst[i] = float2int8(src[i] * s);
    }
}
#endif

#if __AVX512F__
static inline void dynamic_quantize_rowwise_avx512_kernel(const float* src, signed char* dst, float* scale, int width)
{
    __m512 sign_mask = _mm512_set1_ps(-0.f);
    __m512 vmax = _mm512_setzero_ps();
    int i = 0;
    for (; i + 15 < width; i += 16)
    {
        __m512 x = _mm512_loadu_ps(src + i);
        vmax = _mm512_max_ps(vmax, _mm512_andnot_ps(sign_mask, x));
    }
    float absmax = _mm512_comp_reduce_max_ps(vmax);
    for (; i < width; i++)
    {
        absmax = std::max(absmax, fabsf(src[i]));
    }
    float s = absmax == 0.f ? 1.f : 127.f / absmax;
    *scale = s;
    __m512 vscale = _mm512_set1_ps(s);
    i = 0;
    for (; i + 15 < width; i += 16)
    {
        __m512 x = _mm512_loadu_ps(src + i);
        __m512i y = _mm512_cvtps_epi32(_mm512_mul_ps(x, vscale));
        __m128i a0 = _mm512_extracti32x4_epi32(y, 0);
        __m128i a1 = _mm512_extracti32x4_epi32(y, 1);
        __m128i a2 = _mm512_extracti32x4_epi32(y, 2);
        __m128i a3 = _mm512_extracti32x4_epi32(y, 3);
        __m128i b0 = _mm_packs_epi32(a0, a1);
        __m128i b1 = _mm_packs_epi32(a2, a3);
        __m128i c0 = _mm_packs_epi16(b0, b1);
        _mm_storeu_si128((__m128i*)(dst + i), c0);
    }
    for (; i < width; i++)
    {
        dst[i] = float2int8(src[i] * s);
    }
}
#endif

static void dynamic_quantize_rowwise(const float* src, signed char* dst, float* scale, int width)
{
#if __AVX512F__
    dynamic_quantize_rowwise_avx512_kernel(src, dst, scale, width);
#else
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        dynamic_quantize_rowwise_avx2(src, dst, scale, width);
        return;
    }
#endif
#if __AVX2__
    dynamic_quantize_rowwise_avx2(src, dst, scale, width);
#elif __SSE2__
    dynamic_quantize_rowwise_sse2_kernel(src, dst, scale, width);
#else
    dynamic_quantize_rowwise_scalar_kernel(src, dst, scale, width);
#endif
#endif
}

static inline void reciprocal_scales(float* scales, int num_blocks)
{
    int i = 0;
#if __AVX512F__
    for (; i + 15 < num_blocks; i += 16)
    {
        __m512 v = _mm512_loadu_ps(scales + i);
        _mm512_storeu_ps(scales + i, _mm512_div_ps(_mm512_set1_ps(1.0f), v));
    }
#elif __AVX__
    for (; i + 7 < num_blocks; i += 8)
    {
        __m256 v = _mm256_loadu_ps(scales + i);
        _mm256_storeu_ps(scales + i, _mm256_div_ps(_mm256_set1_ps(1.0f), v));
    }
#elif __SSE2__
    for (; i + 3 < num_blocks; i += 4)
    {
        __m128 v = _mm_loadu_ps(scales + i);
        _mm_storeu_ps(scales + i, _mm_div_ps(_mm_set1_ps(1.0f), v));
    }
#endif
    for (; i < num_blocks; i++)
    {
        scales[i] = 1.0f / scales[i];
    }
}

// =================== Int8 SIMD Kernels ===================

static inline int qk_int8_dot_block_scalar_kernel(const signed char* a, const signed char* b, int len)
{
    int sum = 0;
    for (int i = 0; i < len; i++)
        sum += a[i] * b[i];
    return sum;
}

#if __SSE2__ && !__SSSE3__
// SSE2-compatible int8 sign-extension helpers (SSSE3 provides _mm_cvtepi8_epi16)
static inline __m128i _mm_cvtepi8_epi16_sse2(__m128i x)
{
    __m128i sign = _mm_cmpgt_epi8(_mm_setzero_si128(), x);
    return _mm_unpacklo_epi8(x, sign);
}
#define _mm_cvtepi8_epi16 _mm_cvtepi8_epi16_sse2
#endif
#if __SSE2__ && !__SSE4_1__
static inline __m128i _mm_cvtepi8_epi32_sse2(__m128i x)
{
    __m128i sign = _mm_cmpgt_epi8(_mm_setzero_si128(), x);
    __m128i x16 = _mm_unpacklo_epi8(x, sign);
    __m128i sign16 = _mm_cmpgt_epi16(_mm_setzero_si128(), x16);
    return _mm_unpacklo_epi16(x16, sign16);
}
#define _mm_cvtepi8_epi32 _mm_cvtepi8_epi32_sse2
#endif

#if __SSE2__
static inline int qk_int8_dot_block_sse2_kernel(const signed char* a, const signed char* b, int len)
{
    __m128i sum = _mm_setzero_si128();
    int i = 0;
    for (; i + 15 < len; i += 16)
    {
        __m128i va = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i vb = _mm_loadu_si128((const __m128i*)(b + i));
        __m128i va_lo = _mm_cvtepi8_epi16(va);
        __m128i va_hi = _mm_cvtepi8_epi16(_mm_srli_si128(va, 8));
        __m128i vb_lo = _mm_cvtepi8_epi16(vb);
        __m128i vb_hi = _mm_cvtepi8_epi16(_mm_srli_si128(vb, 8));
        sum = _mm_add_epi32(sum, _mm_madd_epi16(va_lo, vb_lo));
        sum = _mm_add_epi32(sum, _mm_madd_epi16(va_hi, vb_hi));
    }
    int sum_tail = 0;
    for (; i < len; i++)
        sum_tail += a[i] * b[i];
    return _mm_reduce_add_epi32(sum) + sum_tail;
}
#endif // __SSE2__

#if __XOP__
static inline int qk_int8_dot_block_xop_kernel(const signed char* a, const signed char* b, int len)
{
    __m128i sum = _mm_setzero_si128();
    int i = 0;
    for (; i + 15 < len; i += 16)
    {
        __m128i va = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i vb = _mm_loadu_si128((const __m128i*)(b + i));
        __m128i va_lo = _mm_cvtepi8_epi16(va);
        __m128i va_hi = _mm_cvtepi8_epi16(_mm_srli_si128(va, 8));
        __m128i vb_lo = _mm_cvtepi8_epi16(vb);
        __m128i vb_hi = _mm_cvtepi8_epi16(_mm_srli_si128(vb, 8));
        sum = _mm_maccd_epi16(va_lo, vb_lo, sum);
        sum = _mm_maccd_epi16(va_hi, vb_hi, sum);
    }
    int sum_tail = 0;
    for (; i < len; i++)
        sum_tail += a[i] * b[i];
    return _mm_reduce_add_epi32(sum) + sum_tail;
}
#endif // __XOP__

#if __AVX2__
inline int qk_int8_dot_block_avx2_kernel(const signed char* a, const signed char* b, int len)
{
    __m256i sum = _mm256_setzero_si256();
    int i = 0;
    for (; i + 31 < len; i += 32)
    {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
        __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        __m256i vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
        __m256i vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
        sum = _mm256_comp_dpwssd_epi32(sum, va_lo, vb_lo);
        sum = _mm256_comp_dpwssd_epi32(sum, va_hi, vb_hi);
    }
    int sum_tail = 0;
    for (; i < len; i++)
        sum_tail += a[i] * b[i];
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    return _mm_reduce_add_epi32(sum128) + sum_tail;
}
#endif // __AVX2__

#if __AVX512F__
static inline int qk_int8_dot_block_avx512_kernel(const signed char* a, const signed char* b, int len)
{
    __m512i sum = _mm512_setzero_si512();
    int i = 0;
    for (; i + 31 < len; i += 32)
    {
        __m256i va256 = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb256 = _mm256_loadu_si256((const __m256i*)(b + i));
        __m512i va = _mm512_cvtepi8_epi16(va256);
        __m512i vb = _mm512_cvtepi8_epi16(vb256);
        sum = _mm512_comp_dpwssd_epi32(sum, va, vb);
    }
    int sum_tail = 0;
    for (; i < len; i++)
        sum_tail += a[i] * b[i];
    __m256i sum256 = _mm256_add_epi32(_mm512_castsi512_si256(sum), _mm512_extracti32x8_epi32(sum, 1));
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum256), _mm256_extracti128_si256(sum256, 1));
    return _mm_reduce_add_epi32(sum128) + sum_tail;
}
#endif // __AVX512F__

static inline int qk_int8_dot_block(const signed char* a, const signed char* b, int len)
{
#if __AVX512F__
    return qk_int8_dot_block_avx512_kernel(a, b, len);
#else
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        return qk_int8_dot_block_avx2(a, b, len);
    }
#endif
#if __AVX2__
    return qk_int8_dot_block_avx2(a, b, len);
#elif __SSE2__
    return qk_int8_dot_block_sse2_kernel(a, b, len);
#else
    return qk_int8_dot_block_scalar_kernel(a, b, len);
#endif
#endif
}

// ------------------- Decode QK Dot Int8 -------------------

static inline void decode_qk_dot_int8_scalar_kernel(float* s, const signed char* q,
        const signed char* K, const float* qscales, const float* kscales,
        int n_start, int block_n, int d, float scale)
{
    const int num_blocks = (d + 31) / 32;
    for (int j = 0; j < block_n; j++)
    {
        const signed char* kptr = K + (n_start + j) * d;
        const float* ks = kscales + (n_start + j);
        float sum = 0.f;
        for (int b = 0; b < num_blocks; b++)
        {
            int off = b * 32;
            int len = std::min(32, d - off);
            if (len <= 0) continue;
            int block_sum = qk_int8_dot_block_scalar_kernel(q + off, kptr + off, len);
            sum += (float)block_sum / (qscales[0] * ks[0]);
        }
        s[j] = sum * scale;
    }
}

#if __SSE2__
static inline void decode_qk_dot_int8_sse2_kernel(float* s, const signed char* q,
        const signed char* K, const float* qscales, const float* kscales,
        int n_start, int block_n, int d, float scale)
{
    const int num_blocks = (d + 31) / 32;
    int j = 0;
    for (; j + 1 < block_n; j += 2)
    {
        const signed char* k0 = K + (n_start + j + 0) * d;
        const signed char* k1 = K + (n_start + j + 1) * d;
        const float* ks0 = kscales + (n_start + j + 0);
        const float* ks1 = kscales + (n_start + j + 1);

        float sum0 = 0.f, sum1 = 0.f;
        for (int b = 0; b < num_blocks; b++)
        {
            float descale = qscales[0] * ks0[0];
            int off = b * 32;
            int len = std::min(32, d - off);
            if (len <= 0) continue;
            int block_sum = qk_int8_dot_block_sse2_kernel(q + off, k0 + off, len);
            sum0 += (float)block_sum * descale;
        }
        for (int b = 0; b < num_blocks; b++)
        {
            float descale = qscales[0] * ks1[0];
            int off = b * 32;
            int len = std::min(32, d - off);
            if (len <= 0) continue;
            int block_sum = qk_int8_dot_block_sse2_kernel(q + off, k1 + off, len);
            sum1 += (float)block_sum * descale;
        }

        s[j + 0] = sum0 * scale;
        s[j + 1] = sum1 * scale;
    }
    for (; j < block_n; j++)
    {
        const signed char* kptr = K + (n_start + j) * d;
        const float* ks = kscales + (n_start + j);
        float sum = 0.f;
        for (int b = 0; b < num_blocks; b++)
        {
            int off = b * 32;
            int len = std::min(32, d - off);
            if (len <= 0) continue;
            int block_sum = qk_int8_dot_block_scalar_kernel(q + off, kptr + off, len);
            float descale = qscales[0] * ks[0];
            sum += (float)block_sum * descale;
        }
        s[j] = sum * scale;
    }
}
#endif // __SSE2__

#if __XOP__
static inline void decode_qk_dot_int8_xop_kernel(float* s, const signed char* q,
        const signed char* K, const float* qscales, const float* kscales,
        int n_start, int block_n, int d, float scale)
{
    const int num_blocks = (d + 31) / 32;
    int j = 0;
    for (; j + 1 < block_n; j += 2)
    {
        const signed char* k0 = K + (n_start + j + 0) * d;
        const signed char* k1 = K + (n_start + j + 1) * d;
        const float* ks0 = kscales + (n_start + j + 0);
        const float* ks1 = kscales + (n_start + j + 1);

        float sum0 = 0.f, sum1 = 0.f;
        for (int b = 0; b < num_blocks; b++)
        {
            int off = b * 32;
            int len = std::min(32, d - off);
            if (len <= 0) continue;
            int block_sum0 = qk_int8_dot_block_xop_kernel(q + off, k0 + off, len);
            int block_sum1 = qk_int8_dot_block_xop_kernel(q + off, k1 + off, len);
            sum0 += (float)block_sum0 * qscales[0] * ks0[0];
            sum1 += (float)block_sum1 * qscales[0] * ks1[0];
        }

        s[j + 0] = sum0 * scale;
        s[j + 1] = sum1 * scale;
    }
    for (; j < block_n; j++)
    {
        const signed char* kptr = K + (n_start + j) * d;
        const float* ks = kscales + (n_start + j);
        float sum = 0.f;
        for (int b = 0; b < num_blocks; b++)
        {
            int off = b * 32;
            int len = std::min(32, d - off);
            if (len <= 0) continue;
            int block_sum = qk_int8_dot_block_xop_kernel(q + off, kptr + off, len);
            sum += (float)block_sum * qscales[0] * ks[0];
        }
        s[j] = sum * scale;
    }
}
#endif // __XOP__

#if __AVX2__
inline void decode_qk_dot_int8_avx2_kernel(float* s, const signed char* q,
        const signed char* K, const float* qscales, const float* kscales,
        int n_start, int block_n, int d, float scale)
{
    const int num_blocks = (d + 31) / 32;
    int j = 0;
    for (; j + 1 < block_n; j += 2)
    {
        const signed char* k0 = K + (n_start + j + 0) * d;
        const signed char* k1 = K + (n_start + j + 1) * d;
        const float* ks0 = kscales + (n_start + j + 0);
        const float* ks1 = kscales + (n_start + j + 1);

        float sum0 = 0.f, sum1 = 0.f;
        for (int b = 0; b < num_blocks; b++)
        {
            int off = b * 32;
            int len = std::min(32, d - off);
            if (len <= 0) continue;
            int block_sum0 = qk_int8_dot_block_avx2_kernel(q + off, k0 + off, len);
            int block_sum1 = qk_int8_dot_block_avx2_kernel(q + off, k1 + off, len);
            sum0 += (float)block_sum0 / (qscales[0] * ks0[0]);
            sum1 += (float)block_sum1 / (qscales[0] * ks1[0]);
        }

        s[j + 0] = sum0 * scale;
        s[j + 1] = sum1 * scale;
    }
    for (; j < block_n; j++)
    {
        const signed char* kptr = K + (n_start + j) * d;
        const float* ks = kscales + (n_start + j);
        float sum = 0.f;
        for (int b = 0; b < num_blocks; b++)
        {
            int off = b * 32;
            int len = std::min(32, d - off);
            if (len <= 0) continue;
            int block_sum = qk_int8_dot_block_avx2_kernel(q + off, kptr + off, len);
            float descale = qscales[0] * ks[0];
            sum += (float)block_sum * descale;
        }
        s[j] = sum * scale;
    }
}
#endif // __AVX2__

#if __AVX2__
static inline int _mm256_reduce_add_epi32(__m256i v);
#endif

#if __AVXVNNI__
static inline void decode_qk_dot_int8_avxvnni_kernel(float* s, const signed char* q,
        const signed char* K, const float* qscales, const float* kscales,
        int n_start, int block_n, int d, float scale)
{
    const float qscale = qscales[0];
    const int num_blocks_32 = d / 32;

    // Precompute qsum for dpbusd compensation: sum((q+128)*k) = sum(q*k) + 128*sum(k)
    int qsum = 0;
    {
        __m256i qsum_acc = _mm256_setzero_si256();
        const __m256i ones = _mm256_set1_epi8(1);
        for (int b = 0; b < num_blocks_32; b++)
        {
            __m256i q_256 = _mm256_loadu_si256((const __m256i*)(q + b * 32));
            qsum_acc = _mm256_dpbusd_epi32(qsum_acc, ones, q_256);
        }
        qsum = _mm256_reduce_add_epi32(qsum_acc);
        for (int i = num_blocks_32 * 32; i < d; i++)
            qsum += q[i];
    }

    int j = 0;
    for (; j + 3 < block_n; j += 4)
    {
        const signed char* k0 = K + (n_start + j + 0) * d;
        const signed char* k1 = K + (n_start + j + 1) * d;
        const signed char* k2 = K + (n_start + j + 2) * d;
        const signed char* k3 = K + (n_start + j + 3) * d;

        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();
        const __m256i offset128 = _mm256_set1_epi8(128);

        for (int b = 0; b < num_blocks_32; b++)
        {
            int off = b * 32;
            __m256i q_u8 = _mm256_add_epi8(_mm256_loadu_si256((const __m256i*)(q + off)), offset128);

            acc0 = _mm256_dpbusd_epi32(acc0, q_u8, _mm256_loadu_si256((const __m256i*)(k0 + off)));
            acc1 = _mm256_dpbusd_epi32(acc1, q_u8, _mm256_loadu_si256((const __m256i*)(k1 + off)));
            acc2 = _mm256_dpbusd_epi32(acc2, q_u8, _mm256_loadu_si256((const __m256i*)(k2 + off)));
            acc3 = _mm256_dpbusd_epi32(acc3, q_u8, _mm256_loadu_si256((const __m256i*)(k3 + off)));
        }

        int scalar0 = 0, scalar1 = 0, scalar2 = 0, scalar3 = 0;
        for (int i = num_blocks_32 * 32; i < d; i++)
        {
            scalar0 += q[i] * k0[i];
            scalar1 += q[i] * k1[i];
            scalar2 += q[i] * k2[i];
            scalar3 += q[i] * k3[i];
        }

        float sum0 = (float)(_mm256_reduce_add_epi32(acc0) - 128 * qsum + scalar0);
        float sum1 = (float)(_mm256_reduce_add_epi32(acc1) - 128 * qsum + scalar1);
        float sum2 = (float)(_mm256_reduce_add_epi32(acc2) - 128 * qsum + scalar2);
        float sum3 = (float)(_mm256_reduce_add_epi32(acc3) - 128 * qsum + scalar3);

        s[j + 0] = sum0 * qscale * kscales[n_start + j + 0] * scale;
        s[j + 1] = sum1 * qscale * kscales[n_start + j + 1] * scale;
        s[j + 2] = sum2 * qscale * kscales[n_start + j + 2] * scale;
        s[j + 3] = sum3 * qscale * kscales[n_start + j + 3] * scale;
    }

    for (; j < block_n; j++)
    {
        const signed char* kptr = K + (n_start + j) * d;
        __m256i acc = _mm256_setzero_si256();
        const __m256i offset128 = _mm256_set1_epi8(128);
        for (int b = 0; b < num_blocks_32; b++)
        {
            int off = b * 32;
            __m256i q_u8 = _mm256_add_epi8(_mm256_loadu_si256((const __m256i*)(q + off)), offset128);
            acc = _mm256_dpbusd_epi32(acc, q_u8, _mm256_loadu_si256((const __m256i*)(kptr + off)));
        }
        int scalar = 0;
        for (int i = num_blocks_32 * 32; i < d; i++)
            scalar += q[i] * kptr[i];
        float sum = (float)(_mm256_reduce_add_epi32(acc) - 128 * qsum + scalar);
        s[j] = sum * qscale * kscales[n_start + j] * scale;
    }
}
#endif // __AVXVNNI__

#if __AVX512F__
static inline void decode_qk_dot_int8_avx512_kernel(float* s, const signed char* q,
        const signed char* K, const float* qscales, const float* kscales,
        int n_start, int block_n, int d, float scale)
{
    const float qscale = qscales[0];
    const int num_blocks_64 = d / 64;

#if __AVX512VNNI__
    // Precompute qsum for dpbusd compensation: sum((q+128)*k) = sum(q*k) + 128*sum(k)
    int qsum = 0;
    {
        __m512i qsum_acc = _mm512_setzero_si512();
        const __m512i ones = _mm512_set1_epi8(1);
        for (int b = 0; b < num_blocks_64; b++)
        {
            __m512i q_512 = _mm512_loadu_si512((const __m512i*)(q + b * 64));
            qsum_acc = _mm512_dpbusd_epi32(qsum_acc, ones, q_512);
        }
        qsum = _mm512_reduce_add_epi32(qsum_acc);
        for (int i = num_blocks_64 * 64; i < d; i++)
            qsum += q[i];
    }
#endif

    int j = 0;
    for (; j + 3 < block_n; j += 4)
    {
        const signed char* k0 = K + (n_start + j + 0) * d;
        const signed char* k1 = K + (n_start + j + 1) * d;
        const signed char* k2 = K + (n_start + j + 2) * d;
        const signed char* k3 = K + (n_start + j + 3) * d;

#if __AVX512VNNI__
        __m512i acc0 = _mm512_setzero_si512();
        __m512i acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512();
        __m512i acc3 = _mm512_setzero_si512();
        const __m512i offset128 = _mm512_set1_epi8(128);

        for (int b = 0; b < num_blocks_64; b++)
        {
            int off = b * 64;
            __m512i q_u8 = _mm512_add_epi8(_mm512_loadu_si512((const __m512i*)(q + off)), offset128);

            acc0 = _mm512_dpbusd_epi32(acc0, q_u8, _mm512_loadu_si512((const __m512i*)(k0 + off)));
            acc1 = _mm512_dpbusd_epi32(acc1, q_u8, _mm512_loadu_si512((const __m512i*)(k1 + off)));
            acc2 = _mm512_dpbusd_epi32(acc2, q_u8, _mm512_loadu_si512((const __m512i*)(k2 + off)));
            acc3 = _mm512_dpbusd_epi32(acc3, q_u8, _mm512_loadu_si512((const __m512i*)(k3 + off)));
        }

        int scalar0 = 0, scalar1 = 0, scalar2 = 0, scalar3 = 0;
        for (int i = num_blocks_64 * 64; i < d; i++)
        {
            scalar0 += q[i] * k0[i];
            scalar1 += q[i] * k1[i];
            scalar2 += q[i] * k2[i];
            scalar3 += q[i] * k3[i];
        }

        float sum0 = (float)(_mm512_reduce_add_epi32(acc0) - 128 * qsum + scalar0);
        float sum1 = (float)(_mm512_reduce_add_epi32(acc1) - 128 * qsum + scalar1);
        float sum2 = (float)(_mm512_reduce_add_epi32(acc2) - 128 * qsum + scalar2);
        float sum3 = (float)(_mm512_reduce_add_epi32(acc3) - 128 * qsum + scalar3);
#else
        __m512i acc0 = _mm512_setzero_epi32();
        __m512i acc1 = _mm512_setzero_epi32();
        __m512i acc2 = _mm512_setzero_epi32();
        __m512i acc3 = _mm512_setzero_epi32();
        int scalar0 = 0, scalar1 = 0, scalar2 = 0, scalar3 = 0;

        for (int b = 0; b < num_blocks_64; b++)
        {
            int off = b * 64;
            __m256i q_32 = _mm256_loadu_si256((const __m256i*)(q + off));
            __m512i q_512 = _mm512_cvtepi8_epi16(q_32);

            acc0 = _mm512_comp_dpwssd_epi32(acc0, q_512, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(k0 + off))));
            acc1 = _mm512_comp_dpwssd_epi32(acc1, q_512, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(k1 + off))));
            acc2 = _mm512_comp_dpwssd_epi32(acc2, q_512, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(k2 + off))));
            acc3 = _mm512_comp_dpwssd_epi32(acc3, q_512, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(k3 + off))));
        }
        for (int i = num_blocks_64 * 64; i < d; i++)
        {
            scalar0 += q[i] * k0[i];
            scalar1 += q[i] * k1[i];
            scalar2 += q[i] * k2[i];
            scalar3 += q[i] * k3[i];
        }

        float sum0 = (float)(_mm512_reduce_add_epi32(acc0) + scalar0);
        float sum1 = (float)(_mm512_reduce_add_epi32(acc1) + scalar1);
        float sum2 = (float)(_mm512_reduce_add_epi32(acc2) + scalar2);
        float sum3 = (float)(_mm512_reduce_add_epi32(acc3) + scalar3);
#endif
        s[j + 0] = sum0 * qscale * kscales[n_start + j + 0] * scale;
        s[j + 1] = sum1 * qscale * kscales[n_start + j + 1] * scale;
        s[j + 2] = sum2 * qscale * kscales[n_start + j + 2] * scale;
        s[j + 3] = sum3 * qscale * kscales[n_start + j + 3] * scale;
    }

    for (; j < block_n; j++)
    {
        const signed char* kptr = K + (n_start + j) * d;
#if __AVX512VNNI__
        __m512i acc = _mm512_setzero_si512();
        const __m512i offset128 = _mm512_set1_epi8(128);
        for (int b = 0; b < num_blocks_64; b++)
        {
            int off = b * 64;
            __m512i q_u8 = _mm512_add_epi8(_mm512_loadu_si512((const __m512i*)(q + off)), offset128);
            acc = _mm512_dpbusd_epi32(acc, q_u8, _mm512_loadu_si512((const __m512i*)(kptr + off)));
        }
        int scalar = 0;
        for (int i = num_blocks_64 * 64; i < d; i++)
            scalar += q[i] * kptr[i];
        float sum = (float)(_mm512_reduce_add_epi32(acc) - 128 * qsum + scalar);
#else
        __m512i acc = _mm512_setzero_epi32();
        int scalar = 0;
        for (int b = 0; b < num_blocks_64; b++)
        {
            int off = b * 64;
            acc = _mm512_comp_dpwssd_epi32(acc,
                                           _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(q + off))),
                                           _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(kptr + off))));
        }
        for (int i = num_blocks_64 * 64; i < d; i++)
            scalar += q[i] * kptr[i];
        float sum = (float)(_mm512_reduce_add_epi32(acc) + scalar);
#endif
        s[j] = sum * qscale * kscales[n_start + j] * scale;
    }
}
#endif // __AVX512F__

static void decode_qk_dot_int8(float* s, const signed char* q,
                               const signed char* K, const float* qscales, const float* kscales,
                               int n_start, int block_n, int d, float scale)
{
#if __AVX512F__
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        decode_qk_dot_int8_avx512vnni(s, q, K, qscales, kscales, n_start, block_n, d, scale);
        return;
    }
#endif
    decode_qk_dot_int8_avx512_kernel(s, q, K, qscales, kscales, n_start, block_n, d, scale);
#else
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        decode_qk_dot_int8_avx512vnni(s, q, K, qscales, kscales, n_start, block_n, d, scale);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni_int8())
    {
        decode_qk_dot_int8_avxvnniint8(s, q, K, qscales, kscales, n_start, block_n, d, scale);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        decode_qk_dot_int8_avxvnni(s, q, K, qscales, kscales, n_start, block_n, d, scale);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        decode_qk_dot_int8_avx2(s, q, K, qscales, kscales, n_start, block_n, d, scale);
        return;
    }
#endif
#if __AVX2__
    decode_qk_dot_int8_avx2_kernel(s, q, K, qscales, kscales, n_start, block_n, d, scale);
#elif __SSE2__
    decode_qk_dot_int8_sse2_kernel(s, q, K, qscales, kscales, n_start, block_n, d, scale);
#else
    decode_qk_dot_int8_scalar_kernel(s, q, K, qscales, kscales, n_start, block_n, d, scale);
#endif
#endif
}

// ------------------- Prefill QK Int8 GEMM Row-wise -------------------

static inline void qk_int8_gemm_row_scalar_kernel(float* s_row,
        const signed char* q_row, const signed char* K, float qscale, const float* kscales,
        int n, int d, float scale)
{
    const int num_blocks = (d + 31) / 32;
    for (int j = 0; j < n; j++)
    {
        const signed char* kptr = K + j * d;
        int sum = 0;
        for (int b = 0; b < num_blocks; b++)
        {
            int off = b * 32;
            int len = std::min(32, d - off);
            if (len <= 0) continue;
            sum += qk_int8_dot_block_scalar_kernel(q_row + off, kptr + off, len);
        }
        s_row[j] = (float)sum * qscale * kscales[j] * scale;
    }
}

#if __SSE2__
static inline void qk_int8_gemm_row_sse2_kernel(float* s_row,
        const signed char* q_row, const signed char* K, float qscale, const float* kscales,
        int n, int d, float scale)
{
    const int num_blocks = (d + 31) / 32;
    int j = 0;
    for (; j + 1 < n; j += 2)
    {
        const signed char* k0 = K + j * d;
        const signed char* k1 = K + (j + 1) * d;

        int sum0 = 0, sum1 = 0;
        for (int b = 0; b < num_blocks; b++)
        {
            int off = b * 32;
            int len = std::min(32, d - off);
            if (len <= 0) continue;
            sum0 += qk_int8_dot_block_sse2_kernel(q_row + off, k0 + off, len);
            sum1 += qk_int8_dot_block_sse2_kernel(q_row + off, k1 + off, len);
        }
        s_row[j] = (float)sum0 * qscale * kscales[j] * scale;
        s_row[j + 1] = (float)sum1 * qscale * kscales[j + 1] * scale;
    }
    for (; j < n; j++)
    {
        const signed char* kptr = K + j * d;
        int sum = 0;
        for (int b = 0; b < num_blocks; b++)
        {
            int off = b * 32;
            int len = std::min(32, d - off);
            if (len <= 0) continue;
            sum += qk_int8_dot_block_scalar_kernel(q_row + off, kptr + off, len);
        }
        s_row[j] = (float)sum * qscale * kscales[j] * scale;
    }
}
#endif // __SSE2__

#if __AVX2__
inline void qk_int8_gemm_row_avx2_kernel(float* s_row,
        const signed char* q_row, const signed char* K, float qscale, const float* kscales,
        int n, int d, float scale)
{
    const int num_blocks = (d + 31) / 32;
    int j = 0;
    for (; j + 1 < n; j += 2)
    {
        const signed char* k0 = K + j * d;
        const signed char* k1 = K + (j + 1) * d;

        int sum0 = 0, sum1 = 0;
        for (int b = 0; b < num_blocks; b++)
        {
            int off = b * 32;
            int len = std::min(32, d - off);
            if (len <= 0) continue;
            sum0 += qk_int8_dot_block_avx2_kernel(q_row + off, k0 + off, len);
            sum1 += qk_int8_dot_block_avx2_kernel(q_row + off, k1 + off, len);
        }
        s_row[j] = (float)sum0 * qscale * kscales[j] * scale;
        s_row[j + 1] = (float)sum1 * qscale * kscales[j + 1] * scale;
    }
    for (; j < n; j++)
    {
        const signed char* kptr = K + j * d;
        int sum = 0;
        for (int b = 0; b < num_blocks; b++)
        {
            int off = b * 32;
            int len = std::min(32, d - off);
            if (len <= 0) continue;
            sum += qk_int8_dot_block_avx2_kernel(q_row + off, kptr + off, len);
        }
        s_row[j] = (float)sum * qscale * kscales[j] * scale;
    }
}
#endif // __AVX2__

#if __AVX512F__
static inline void qk_int8_gemm_row_avx512_kernel(float* s_row,
        const signed char* q_row, const signed char* K, float qscale, const float* kscales,
        int n, int d, float scale)
{
    const int num_blocks = (d + 31) / 32;
    int j = 0;
    for (; j + 3 < n; j += 4)
    {
        const signed char* k0 = K + j * d;
        const signed char* k1 = K + (j + 1) * d;
        const signed char* k2 = K + (j + 2) * d;
        const signed char* k3 = K + (j + 3) * d;

        __m512i acc0 = _mm512_setzero_epi32();
        __m512i acc1 = _mm512_setzero_epi32();
        __m512i acc2 = _mm512_setzero_epi32();
        __m512i acc3 = _mm512_setzero_epi32();
        int scalar0 = 0, scalar1 = 0, scalar2 = 0, scalar3 = 0;
        for (int b = 0; b < num_blocks; b++)
        {
            int off = b * 32;
            int len = std::min(32, d - off);
            if (len <= 0) continue;
            if (len == 32)
            {
                __m256i q_32 = _mm256_loadu_si256((const __m256i*)(q_row + off));
                __m512i q_512 = _mm512_cvtepi8_epi16(q_32);

                __m256i k0_32 = _mm256_loadu_si256((const __m256i*)(k0 + off));
                __m256i k1_32 = _mm256_loadu_si256((const __m256i*)(k1 + off));
                __m256i k2_32 = _mm256_loadu_si256((const __m256i*)(k2 + off));
                __m256i k3_32 = _mm256_loadu_si256((const __m256i*)(k3 + off));
                __m512i k0_512 = _mm512_cvtepi8_epi16(k0_32);
                __m512i k1_512 = _mm512_cvtepi8_epi16(k1_32);
                __m512i k2_512 = _mm512_cvtepi8_epi16(k2_32);
                __m512i k3_512 = _mm512_cvtepi8_epi16(k3_32);

                acc0 = _mm512_comp_dpwssd_epi32(acc0, q_512, k0_512);
                acc1 = _mm512_comp_dpwssd_epi32(acc1, q_512, k1_512);
                acc2 = _mm512_comp_dpwssd_epi32(acc2, q_512, k2_512);
                acc3 = _mm512_comp_dpwssd_epi32(acc3, q_512, k3_512);
            }
            else
            {
                int bs;
                bs = qk_int8_dot_block_avx512_kernel(q_row + off, k0 + off, len);
                scalar0 += bs;
                bs = qk_int8_dot_block_avx512_kernel(q_row + off, k1 + off, len);
                scalar1 += bs;
                bs = qk_int8_dot_block_avx512_kernel(q_row + off, k2 + off, len);
                scalar2 += bs;
                bs = qk_int8_dot_block_avx512_kernel(q_row + off, k3 + off, len);
                scalar3 += bs;
            }
        }
        float descale0 = qscale * kscales[j];
        float descale1 = qscale * kscales[j + 1];
        float descale2 = qscale * kscales[j + 2];
        float descale3 = qscale * kscales[j + 3];
        float sum0 = (float)_mm512_reduce_add_epi32(acc0) + (float)scalar0;
        float sum1 = (float)_mm512_reduce_add_epi32(acc1) + (float)scalar1;
        float sum2 = (float)_mm512_reduce_add_epi32(acc2) + (float)scalar2;
        float sum3 = (float)_mm512_reduce_add_epi32(acc3) + (float)scalar3;
        s_row[j] = sum0 * descale0 * scale;
        s_row[j + 1] = sum1 * descale1 * scale;
        s_row[j + 2] = sum2 * descale2 * scale;
        s_row[j + 3] = sum3 * descale3 * scale;
    }
    for (; j < n; j++)
    {
        const signed char* kptr = K + j * d;
        __m512i acc = _mm512_setzero_epi32();
        int scalar_sum = 0;
        for (int b = 0; b < num_blocks; b++)
        {
            int off = b * 32;
            int len = std::min(32, d - off);
            if (len <= 0) continue;
            if (len == 32)
            {
                __m256i q_32 = _mm256_loadu_si256((const __m256i*)(q_row + off));
                __m256i k_32 = _mm256_loadu_si256((const __m256i*)(kptr + off));
                __m512i q_512 = _mm512_cvtepi8_epi16(q_32);
                __m512i k_512 = _mm512_cvtepi8_epi16(k_32);
                acc = _mm512_comp_dpwssd_epi32(acc, q_512, k_512);
            }
            else
            {
                scalar_sum += qk_int8_dot_block_avx512_kernel(q_row + off, kptr + off, len);
            }
        }
        float sum = (float)_mm512_reduce_add_epi32(acc) + (float)scalar_sum;
        s_row[j] = sum * qscale * kscales[j] * scale;
    }
}
#endif // __AVX512F__

#if __AVX512VNNI__
static inline void qk_int8_gemm_row_avx512vnni_kernel(float* s_row,
        const signed char* q_row, const signed char* K, float qscale, const float* kscales,
        int n, int d, float scale)
{
    const int num_blocks_64 = d / 64;
    int qsum_64byte = 0;
    if (num_blocks_64 > 0)
    {
        __m512i ones = _mm512_set1_epi8(1);
        __m512i sum_acc = _mm512_setzero_epi32();
        for (int b = 0; b < num_blocks_64; b++)
        {
            __m512i q_512 = _mm512_loadu_si512((const __m512i*)(q_row + b * 64));
            sum_acc = _mm512_dpbusd_epi32(sum_acc, ones, q_512);
        }
        qsum_64byte = _mm512_reduce_add_epi32(sum_acc);
    }

    int j = 0;
    for (; j + 3 < n; j += 4)
    {
        const signed char* k0 = K + j * d;
        const signed char* k1 = K + (j + 1) * d;
        const signed char* k2 = K + (j + 2) * d;
        const signed char* k3 = K + (j + 3) * d;

        __m512i acc0 = _mm512_setzero_epi32();
        __m512i acc1 = _mm512_setzero_epi32();
        __m512i acc2 = _mm512_setzero_epi32();
        __m512i acc3 = _mm512_setzero_epi32();
        int scalar0 = 0, scalar1 = 0, scalar2 = 0, scalar3 = 0;

        for (int b = 0; b < num_blocks_64; b++)
        {
            int off = b * 64;
            __m512i q_512 = _mm512_loadu_si512((const __m512i*)(q_row + off));
            __m512i k0_512 = _mm512_loadu_si512((const __m512i*)(k0 + off));
            __m512i k1_512 = _mm512_loadu_si512((const __m512i*)(k1 + off));
            __m512i k2_512 = _mm512_loadu_si512((const __m512i*)(k2 + off));
            __m512i k3_512 = _mm512_loadu_si512((const __m512i*)(k3 + off));
            __m512i k0_u8 = _mm512_add_epi8(k0_512, _mm512_set1_epi8(128));
            __m512i k1_u8 = _mm512_add_epi8(k1_512, _mm512_set1_epi8(128));
            __m512i k2_u8 = _mm512_add_epi8(k2_512, _mm512_set1_epi8(128));
            __m512i k3_u8 = _mm512_add_epi8(k3_512, _mm512_set1_epi8(128));

            acc0 = _mm512_dpbusd_epi32(acc0, k0_u8, q_512);
            acc1 = _mm512_dpbusd_epi32(acc1, k1_u8, q_512);
            acc2 = _mm512_dpbusd_epi32(acc2, k2_u8, q_512);
            acc3 = _mm512_dpbusd_epi32(acc3, k3_u8, q_512);
        }

        int tail_start = num_blocks_64 * 64;
        if (tail_start < d)
        {
            for (int k = tail_start; k < d; k++)
            {
                scalar0 += q_row[k] * k0[k];
                scalar1 += q_row[k] * k1[k];
                scalar2 += q_row[k] * k2[k];
                scalar3 += q_row[k] * k3[k];
            }
        }

        float descale0 = qscale * kscales[j];
        float descale1 = qscale * kscales[j + 1];
        float descale2 = qscale * kscales[j + 2];
        float descale3 = qscale * kscales[j + 3];
        float sum0 = (float)(_mm512_reduce_add_epi32(acc0) - 128 * qsum_64byte) + (float)scalar0;
        float sum1 = (float)(_mm512_reduce_add_epi32(acc1) - 128 * qsum_64byte) + (float)scalar1;
        float sum2 = (float)(_mm512_reduce_add_epi32(acc2) - 128 * qsum_64byte) + (float)scalar2;
        float sum3 = (float)(_mm512_reduce_add_epi32(acc3) - 128 * qsum_64byte) + (float)scalar3;
        s_row[j] = sum0 * descale0 * scale;
        s_row[j + 1] = sum1 * descale1 * scale;
        s_row[j + 2] = sum2 * descale2 * scale;
        s_row[j + 3] = sum3 * descale3 * scale;
    }
    for (; j < n; j++)
    {
        const signed char* kptr = K + j * d;
        __m512i acc = _mm512_setzero_epi32();
        int scalar_sum = 0;
        for (int b = 0; b < num_blocks_64; b++)
        {
            int off = b * 64;
            __m512i q_512 = _mm512_loadu_si512((const __m512i*)(q_row + off));
            __m512i k_512 = _mm512_loadu_si512((const __m512i*)(kptr + off));
            __m512i k_u8 = _mm512_add_epi8(k_512, _mm512_set1_epi8(128));
            acc = _mm512_dpbusd_epi32(acc, k_u8, q_512);
        }
        int tail_start = num_blocks_64 * 64;
        if (tail_start < d)
        {
            for (int k = tail_start; k < d; k++)
            {
                scalar_sum += q_row[k] * kptr[k];
            }
        }
        float sum = (float)(_mm512_reduce_add_epi32(acc) - 128 * qsum_64byte) + (float)scalar_sum;
        s_row[j] = sum * qscale * kscales[j] * scale;
    }
}
#endif // __AVX512VNNI__

static void qk_int8_gemm_row(float* s_row,
                             const signed char* q_row, const signed char* K, float qscale, const float* kscales,
                             int n, int d, float scale)
{
#if __AVX512F__
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        qk_int8_gemm_row_avx512vnni(s_row, q_row, K, qscale, kscales, n, d, scale);
        return;
    }
#endif
#if __AVX512VNNI__
    qk_int8_gemm_row_avx512vnni_kernel(s_row, q_row, K, qscale, kscales, n, d, scale);
#else
    qk_int8_gemm_row_avx512_kernel(s_row, q_row, K, qscale, kscales, n, d, scale);
#endif
#else
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        qk_int8_gemm_row_avx512vnni(s_row, q_row, K, qscale, kscales, n, d, scale);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni_int8())
    {
        qk_int8_gemm_row_avxvnniint8(s_row, q_row, K, qscale, kscales, n, d, scale);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        qk_int8_gemm_row_avxvnni(s_row, q_row, K, qscale, kscales, n, d, scale);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        qk_int8_gemm_row_avx2(s_row, q_row, K, qscale, kscales, n, d, scale);
        return;
    }
#endif
#if __AVX2__
    qk_int8_gemm_row_avx2_kernel(s_row, q_row, K, qscale, kscales, n, d, scale);
#elif __SSE2__
    qk_int8_gemm_row_sse2_kernel(s_row, q_row, K, qscale, kscales, n, d, scale);
#else
    qk_int8_gemm_row_scalar_kernel(s_row, q_row, K, qscale, kscales, n, d, scale);
#endif
#endif
}

// ------------------- Tiled QK Int8 GEMM (M-tiling) -------------------

static inline void qk_int8_gemm_tiled_scalar_kernel(float* S,
        const signed char* Q, const signed char* K,
        const float* qscales, const float* kscales,
        int m, int n, int d, float scale)
{
    const int num_blocks = (d + 31) / 32;
    int i = 0;
    for (; i + 2 <= m; i += 2)
    {
        const signed char* q0 = Q + (i + 0) * d;
        const signed char* q1 = Q + (i + 1) * d;
        float qs0 = qscales[i + 0];
        float qs1 = qscales[i + 1];
        for (int j = 0; j < n; j++)
        {
            const signed char* kptr = K + j * d;
            int sum0 = 0, sum1 = 0;
            for (int b = 0; b < num_blocks; b++)
            {
                int off = b * 32;
                int len = std::min(32, d - off);
                if (len <= 0) continue;
                sum0 += qk_int8_dot_block_scalar_kernel(q0 + off, kptr + off, len);
                sum1 += qk_int8_dot_block_scalar_kernel(q1 + off, kptr + off, len);
            }
            S[(i + 0) * n + j] = (float)sum0 * qs0 * kscales[j] * scale;
            S[(i + 1) * n + j] = (float)sum1 * qs1 * kscales[j] * scale;
        }
    }
    for (; i < m; i++)
    {
        qk_int8_gemm_row_scalar_kernel(S + i * n, Q + i * d, K, qscales[i], kscales, n, d, scale);
    }
}

#if __SSE2__
static inline void qk_int8_gemm_tiled_sse2_kernel(float* S,
        const signed char* Q, const signed char* K,
        const float* qscales, const float* kscales,
        int m, int n, int d, float scale)
{
    const int num_blocks = (d + 31) / 32;
    int i = 0;
    for (; i + 2 <= m; i += 2)
    {
        const signed char* q0 = Q + (i + 0) * d;
        const signed char* q1 = Q + (i + 1) * d;
        float qs0 = qscales[i + 0];
        float qs1 = qscales[i + 1];
        int j = 0;
        for (; j + 1 < n; j += 2)
        {
            const signed char* k0 = K + j * d;
            const signed char* k1 = K + (j + 1) * d;
            int sum00 = 0, sum01 = 0, sum10 = 0, sum11 = 0;
            for (int b = 0; b < num_blocks; b++)
            {
                int off = b * 32;
                int len = std::min(32, d - off);
                if (len <= 0) continue;
                int bs;
                bs = qk_int8_dot_block_sse2_kernel(q0 + off, k0 + off, len);
                sum00 += bs;
                bs = qk_int8_dot_block_sse2_kernel(q0 + off, k1 + off, len);
                sum01 += bs;
                bs = qk_int8_dot_block_sse2_kernel(q1 + off, k0 + off, len);
                sum10 += bs;
                bs = qk_int8_dot_block_sse2_kernel(q1 + off, k1 + off, len);
                sum11 += bs;
            }
            S[(i + 0) * n + j] = (float)sum00 * qs0 * kscales[j] * scale;
            S[(i + 0) * n + j + 1] = (float)sum01 * qs0 * kscales[j + 1] * scale;
            S[(i + 1) * n + j] = (float)sum10 * qs1 * kscales[j] * scale;
            S[(i + 1) * n + j + 1] = (float)sum11 * qs1 * kscales[j + 1] * scale;
        }
        for (; j < n; j++)
        {
            const signed char* kptr = K + j * d;
            int sum0 = 0, sum1 = 0;
            for (int b = 0; b < num_blocks; b++)
            {
                int off = b * 32;
                int len = std::min(32, d - off);
                if (len <= 0) continue;
                sum0 += qk_int8_dot_block_sse2_kernel(q0 + off, kptr + off, len);
                sum1 += qk_int8_dot_block_sse2_kernel(q1 + off, kptr + off, len);
            }
            S[(i + 0) * n + j] = (float)sum0 * qs0 * kscales[j] * scale;
            S[(i + 1) * n + j] = (float)sum1 * qs1 * kscales[j] * scale;
        }
    }
    for (; i < m; i++)
    {
        qk_int8_gemm_row_sse2_kernel(S + i * n, Q + i * d, K, qscales[i], kscales, n, d, scale);
    }
}
#endif // __SSE2__

#if __AVX2__
static inline int _mm256_reduce_add_epi32(__m256i v)
{
    __m128i vlow = _mm256_castsi256_si128(v);
    __m128i vhigh = _mm256_extracti128_si256(v, 1);
    vlow = _mm_add_epi32(vlow, vhigh);
    return _mm_reduce_add_epi32(vlow);
}

#if __AVX512F__
static inline float _mm512_reduce_add_ps(__m512 v)
{
    __m256 vlow = _mm512_castps512_ps256(v);
    __m256 vhigh = _mm512_extractf32x8_ps(v, 1);
    return _mm256_reduce_add_ps(_mm256_add_ps(vlow, vhigh));
}

static inline int _mm512_reduce_add_epi32(__m512i v)
{
    __m256i vlow = _mm512_castsi512_si256(v);
    __m256i vhigh = _mm512_extracti32x8_epi32(v, 1);
    return _mm256_reduce_add_epi32(_mm256_add_epi32(vlow, vhigh));
}
#endif // __AVX512F__

inline void qk_int8_gemm_tiled_avx2_kernel(float* S,
        const signed char* Q, const signed char* K,
        const float* qscales, const float* kscales,
        int m, int n, int d, float scale)
{
    const int num_blocks = (d + 31) / 32;
    int i = 0;
    for (; i + 2 <= m; i += 2)
    {
        const signed char* q0 = Q + (i + 0) * d;
        const signed char* q1 = Q + (i + 1) * d;
        float qs0 = qscales[i + 0];
        float qs1 = qscales[i + 1];
        int j = 0;
        for (; j + 3 < n; j += 4)
        {
            const signed char* k0 = K + j * d;
            const signed char* k1 = K + (j + 1) * d;
            const signed char* k2 = K + (j + 2) * d;
            const signed char* k3 = K + (j + 3) * d;

            __m256i acc00 = _mm256_setzero_si256();
            __m256i acc01 = _mm256_setzero_si256();
            __m256i acc02 = _mm256_setzero_si256();
            __m256i acc03 = _mm256_setzero_si256();
            __m256i acc10 = _mm256_setzero_si256();
            __m256i acc11 = _mm256_setzero_si256();
            __m256i acc12 = _mm256_setzero_si256();
            __m256i acc13 = _mm256_setzero_si256();
            int scalar00 = 0, scalar01 = 0, scalar02 = 0, scalar03 = 0;
            int scalar10 = 0, scalar11 = 0, scalar12 = 0, scalar13 = 0;

            for (int b = 0; b < num_blocks; b++)
            {
                int off = b * 32;
                int len = std::min(32, d - off);
                if (len <= 0) continue;
                if (len == 32)
                {
                    __m256i q0_32 = _mm256_loadu_si256((const __m256i*)(q0 + off));
                    __m256i q1_32 = _mm256_loadu_si256((const __m256i*)(q1 + off));
                    __m256i q0_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q0_32));
                    __m256i q0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q0_32, 1));
                    __m256i q1_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q1_32));
                    __m256i q1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q1_32, 1));

                    __m256i k0_32 = _mm256_loadu_si256((const __m256i*)(k0 + off));
                    __m256i k0_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(k0_32));
                    __m256i k0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(k0_32, 1));
                    acc00 = _mm256_comp_dpwssd_epi32(acc00, q0_lo, k0_lo);
                    acc00 = _mm256_comp_dpwssd_epi32(acc00, q0_hi, k0_hi);
                    acc10 = _mm256_comp_dpwssd_epi32(acc10, q1_lo, k0_lo);
                    acc10 = _mm256_comp_dpwssd_epi32(acc10, q1_hi, k0_hi);

                    __m256i k1_32 = _mm256_loadu_si256((const __m256i*)(k1 + off));
                    __m256i k1_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(k1_32));
                    __m256i k1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(k1_32, 1));
                    acc01 = _mm256_comp_dpwssd_epi32(acc01, q0_lo, k1_lo);
                    acc01 = _mm256_comp_dpwssd_epi32(acc01, q0_hi, k1_hi);
                    acc11 = _mm256_comp_dpwssd_epi32(acc11, q1_lo, k1_lo);
                    acc11 = _mm256_comp_dpwssd_epi32(acc11, q1_hi, k1_hi);

                    __m256i k2_32 = _mm256_loadu_si256((const __m256i*)(k2 + off));
                    __m256i k2_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(k2_32));
                    __m256i k2_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(k2_32, 1));
                    acc02 = _mm256_comp_dpwssd_epi32(acc02, q0_lo, k2_lo);
                    acc02 = _mm256_comp_dpwssd_epi32(acc02, q0_hi, k2_hi);
                    acc12 = _mm256_comp_dpwssd_epi32(acc12, q1_lo, k2_lo);
                    acc12 = _mm256_comp_dpwssd_epi32(acc12, q1_hi, k2_hi);

                    __m256i k3_32 = _mm256_loadu_si256((const __m256i*)(k3 + off));
                    __m256i k3_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(k3_32));
                    __m256i k3_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(k3_32, 1));
                    acc03 = _mm256_comp_dpwssd_epi32(acc03, q0_lo, k3_lo);
                    acc03 = _mm256_comp_dpwssd_epi32(acc03, q0_hi, k3_hi);
                    acc13 = _mm256_comp_dpwssd_epi32(acc13, q1_lo, k3_lo);
                    acc13 = _mm256_comp_dpwssd_epi32(acc13, q1_hi, k3_hi);
                }
                else
                {
                    int bs;
                    bs = qk_int8_dot_block_avx2_kernel(q0 + off, k0 + off, len);
                    scalar00 += bs;
                    bs = qk_int8_dot_block_avx2_kernel(q0 + off, k1 + off, len);
                    scalar01 += bs;
                    bs = qk_int8_dot_block_avx2_kernel(q0 + off, k2 + off, len);
                    scalar02 += bs;
                    bs = qk_int8_dot_block_avx2_kernel(q0 + off, k3 + off, len);
                    scalar03 += bs;
                    bs = qk_int8_dot_block_avx2_kernel(q1 + off, k0 + off, len);
                    scalar10 += bs;
                    bs = qk_int8_dot_block_avx2_kernel(q1 + off, k1 + off, len);
                    scalar11 += bs;
                    bs = qk_int8_dot_block_avx2_kernel(q1 + off, k2 + off, len);
                    scalar12 += bs;
                    bs = qk_int8_dot_block_avx2_kernel(q1 + off, k3 + off, len);
                    scalar13 += bs;
                }
            }
            float descale00 = qs0 * kscales[j];
            float descale01 = qs0 * kscales[j + 1];
            float descale02 = qs0 * kscales[j + 2];
            float descale03 = qs0 * kscales[j + 3];
            float descale10 = qs1 * kscales[j];
            float descale11 = qs1 * kscales[j + 1];
            float descale12 = qs1 * kscales[j + 2];
            float descale13 = qs1 * kscales[j + 3];
            S[(i + 0) * n + j] = ((float)_mm256_reduce_add_epi32(acc00) + (float)scalar00) * descale00 * scale;
            S[(i + 0) * n + j + 1] = ((float)_mm256_reduce_add_epi32(acc01) + (float)scalar01) * descale01 * scale;
            S[(i + 0) * n + j + 2] = ((float)_mm256_reduce_add_epi32(acc02) + (float)scalar02) * descale02 * scale;
            S[(i + 0) * n + j + 3] = ((float)_mm256_reduce_add_epi32(acc03) + (float)scalar03) * descale03 * scale;
            S[(i + 1) * n + j] = ((float)_mm256_reduce_add_epi32(acc10) + (float)scalar10) * descale10 * scale;
            S[(i + 1) * n + j + 1] = ((float)_mm256_reduce_add_epi32(acc11) + (float)scalar11) * descale11 * scale;
            S[(i + 1) * n + j + 2] = ((float)_mm256_reduce_add_epi32(acc12) + (float)scalar12) * descale12 * scale;
            S[(i + 1) * n + j + 3] = ((float)_mm256_reduce_add_epi32(acc13) + (float)scalar13) * descale13 * scale;
        }
        for (; j + 1 < n; j += 2)
        {
            const signed char* k0 = K + j * d;
            const signed char* k1 = K + (j + 1) * d;
            int sum00 = 0, sum01 = 0, sum10 = 0, sum11 = 0;
            for (int b = 0; b < num_blocks; b++)
            {
                int off = b * 32;
                int len = std::min(32, d - off);
                if (len <= 0) continue;
                int bs;
                bs = qk_int8_dot_block_avx2_kernel(q0 + off, k0 + off, len);
                sum00 += bs;
                bs = qk_int8_dot_block_avx2_kernel(q0 + off, k1 + off, len);
                sum01 += bs;
                bs = qk_int8_dot_block_avx2_kernel(q1 + off, k0 + off, len);
                sum10 += bs;
                bs = qk_int8_dot_block_avx2_kernel(q1 + off, k1 + off, len);
                sum11 += bs;
            }
            S[(i + 0) * n + j] = (float)sum00 * qs0 * kscales[j] * scale;
            S[(i + 0) * n + j + 1] = (float)sum01 * qs0 * kscales[j + 1] * scale;
            S[(i + 1) * n + j] = (float)sum10 * qs1 * kscales[j] * scale;
            S[(i + 1) * n + j + 1] = (float)sum11 * qs1 * kscales[j + 1] * scale;
        }
        for (; j < n; j++)
        {
            const signed char* kptr = K + j * d;
            int sum0 = 0, sum1 = 0;
            for (int b = 0; b < num_blocks; b++)
            {
                int off = b * 32;
                int len = std::min(32, d - off);
                if (len <= 0) continue;
                sum0 += qk_int8_dot_block_avx2_kernel(q0 + off, kptr + off, len);
                sum1 += qk_int8_dot_block_avx2_kernel(q1 + off, kptr + off, len);
            }
            S[(i + 0) * n + j] = (float)sum0 * qs0 * kscales[j] * scale;
            S[(i + 1) * n + j] = (float)sum1 * qs1 * kscales[j] * scale;
        }
    }
    for (; i < m; i++)
    {
        qk_int8_gemm_row_avx2_kernel(S + i * n, Q + i * d, K, qscales[i], kscales, n, d, scale);
    }
}
#endif // __AVX2__

#if __AVX512F__
static inline void qk_int8_gemm_tiled_avx512_kernel(float* S,
        const signed char* Q, const signed char* K,
        const float* qscales, const float* kscales,
        int m, int n, int d, float scale)
{
    const int num_blocks = (d + 31) / 32;
    int i = 0;
    for (; i + 4 <= m; i += 4)
    {
        const signed char* q0 = Q + (i + 0) * d;
        const signed char* q1 = Q + (i + 1) * d;
        const signed char* q2 = Q + (i + 2) * d;
        const signed char* q3 = Q + (i + 3) * d;
        float qs0 = qscales[i + 0];
        float qs1 = qscales[i + 1];
        float qs2 = qscales[i + 2];
        float qs3 = qscales[i + 3];
        int j = 0;
        for (; j + 3 < n; j += 4)
        {
            const signed char* k0 = K + j * d;
            const signed char* k1 = K + (j + 1) * d;
            const signed char* k2 = K + (j + 2) * d;
            const signed char* k3 = K + (j + 3) * d;

            __m512i acc00 = _mm512_setzero_epi32();
            __m512i acc01 = _mm512_setzero_epi32();
            __m512i acc02 = _mm512_setzero_epi32();
            __m512i acc03 = _mm512_setzero_epi32();
            __m512i acc10 = _mm512_setzero_epi32();
            __m512i acc11 = _mm512_setzero_epi32();
            __m512i acc12 = _mm512_setzero_epi32();
            __m512i acc13 = _mm512_setzero_epi32();
            __m512i acc20 = _mm512_setzero_epi32();
            __m512i acc21 = _mm512_setzero_epi32();
            __m512i acc22 = _mm512_setzero_epi32();
            __m512i acc23 = _mm512_setzero_epi32();
            __m512i acc30 = _mm512_setzero_epi32();
            __m512i acc31 = _mm512_setzero_epi32();
            __m512i acc32 = _mm512_setzero_epi32();
            __m512i acc33 = _mm512_setzero_epi32();
            int scalar00 = 0, scalar01 = 0, scalar02 = 0, scalar03 = 0;
            int scalar10 = 0, scalar11 = 0, scalar12 = 0, scalar13 = 0;
            int scalar20 = 0, scalar21 = 0, scalar22 = 0, scalar23 = 0;
            int scalar30 = 0, scalar31 = 0, scalar32 = 0, scalar33 = 0;

            for (int b = 0; b < num_blocks; b++)
            {
                int off = b * 32;
                int len = std::min(32, d - off);
                if (len <= 0) continue;
                if (len == 32)
                {
                    __m256i q0_32 = _mm256_loadu_si256((const __m256i*)(q0 + off));
                    __m256i q1_32 = _mm256_loadu_si256((const __m256i*)(q1 + off));
                    __m256i q2_32 = _mm256_loadu_si256((const __m256i*)(q2 + off));
                    __m256i q3_32 = _mm256_loadu_si256((const __m256i*)(q3 + off));
                    __m512i q0_512 = _mm512_cvtepi8_epi16(q0_32);
                    __m512i q1_512 = _mm512_cvtepi8_epi16(q1_32);
                    __m512i q2_512 = _mm512_cvtepi8_epi16(q2_32);
                    __m512i q3_512 = _mm512_cvtepi8_epi16(q3_32);

                    __m256i k0_32 = _mm256_loadu_si256((const __m256i*)(k0 + off));
                    __m256i k1_32 = _mm256_loadu_si256((const __m256i*)(k1 + off));
                    __m256i k2_32 = _mm256_loadu_si256((const __m256i*)(k2 + off));
                    __m256i k3_32 = _mm256_loadu_si256((const __m256i*)(k3 + off));
                    __m512i k0_512 = _mm512_cvtepi8_epi16(k0_32);
                    __m512i k1_512 = _mm512_cvtepi8_epi16(k1_32);
                    __m512i k2_512 = _mm512_cvtepi8_epi16(k2_32);
                    __m512i k3_512 = _mm512_cvtepi8_epi16(k3_32);

                    acc00 = _mm512_comp_dpwssd_epi32(acc00, q0_512, k0_512);
                    acc01 = _mm512_comp_dpwssd_epi32(acc01, q0_512, k1_512);
                    acc02 = _mm512_comp_dpwssd_epi32(acc02, q0_512, k2_512);
                    acc03 = _mm512_comp_dpwssd_epi32(acc03, q0_512, k3_512);
                    acc10 = _mm512_comp_dpwssd_epi32(acc10, q1_512, k0_512);
                    acc11 = _mm512_comp_dpwssd_epi32(acc11, q1_512, k1_512);
                    acc12 = _mm512_comp_dpwssd_epi32(acc12, q1_512, k2_512);
                    acc13 = _mm512_comp_dpwssd_epi32(acc13, q1_512, k3_512);
                    acc20 = _mm512_comp_dpwssd_epi32(acc20, q2_512, k0_512);
                    acc21 = _mm512_comp_dpwssd_epi32(acc21, q2_512, k1_512);
                    acc22 = _mm512_comp_dpwssd_epi32(acc22, q2_512, k2_512);
                    acc23 = _mm512_comp_dpwssd_epi32(acc23, q2_512, k3_512);
                    acc30 = _mm512_comp_dpwssd_epi32(acc30, q3_512, k0_512);
                    acc31 = _mm512_comp_dpwssd_epi32(acc31, q3_512, k1_512);
                    acc32 = _mm512_comp_dpwssd_epi32(acc32, q3_512, k2_512);
                    acc33 = _mm512_comp_dpwssd_epi32(acc33, q3_512, k3_512);
                }
                else
                {
                    int bs;
                    bs = qk_int8_dot_block_avx512_kernel(q0 + off, k0 + off, len);
                    scalar00 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q0 + off, k1 + off, len);
                    scalar01 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q0 + off, k2 + off, len);
                    scalar02 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q0 + off, k3 + off, len);
                    scalar03 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q1 + off, k0 + off, len);
                    scalar10 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q1 + off, k1 + off, len);
                    scalar11 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q1 + off, k2 + off, len);
                    scalar12 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q1 + off, k3 + off, len);
                    scalar13 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q2 + off, k0 + off, len);
                    scalar20 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q2 + off, k1 + off, len);
                    scalar21 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q2 + off, k2 + off, len);
                    scalar22 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q2 + off, k3 + off, len);
                    scalar23 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q3 + off, k0 + off, len);
                    scalar30 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q3 + off, k1 + off, len);
                    scalar31 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q3 + off, k2 + off, len);
                    scalar32 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q3 + off, k3 + off, len);
                    scalar33 += bs;
                }
            }
            float descale00 = qs0 * kscales[j];
            float descale01 = qs0 * kscales[j + 1];
            float descale02 = qs0 * kscales[j + 2];
            float descale03 = qs0 * kscales[j + 3];
            float descale10 = qs1 * kscales[j];
            float descale11 = qs1 * kscales[j + 1];
            float descale12 = qs1 * kscales[j + 2];
            float descale13 = qs1 * kscales[j + 3];
            float descale20 = qs2 * kscales[j];
            float descale21 = qs2 * kscales[j + 1];
            float descale22 = qs2 * kscales[j + 2];
            float descale23 = qs2 * kscales[j + 3];
            float descale30 = qs3 * kscales[j];
            float descale31 = qs3 * kscales[j + 1];
            float descale32 = qs3 * kscales[j + 2];
            float descale33 = qs3 * kscales[j + 3];
            float sum00 = (float)_mm512_reduce_add_epi32(acc00) + (float)scalar00;
            float sum01 = (float)_mm512_reduce_add_epi32(acc01) + (float)scalar01;
            float sum02 = (float)_mm512_reduce_add_epi32(acc02) + (float)scalar02;
            float sum03 = (float)_mm512_reduce_add_epi32(acc03) + (float)scalar03;
            float sum10 = (float)_mm512_reduce_add_epi32(acc10) + (float)scalar10;
            float sum11 = (float)_mm512_reduce_add_epi32(acc11) + (float)scalar11;
            float sum12 = (float)_mm512_reduce_add_epi32(acc12) + (float)scalar12;
            float sum13 = (float)_mm512_reduce_add_epi32(acc13) + (float)scalar13;
            float sum20 = (float)_mm512_reduce_add_epi32(acc20) + (float)scalar20;
            float sum21 = (float)_mm512_reduce_add_epi32(acc21) + (float)scalar21;
            float sum22 = (float)_mm512_reduce_add_epi32(acc22) + (float)scalar22;
            float sum23 = (float)_mm512_reduce_add_epi32(acc23) + (float)scalar23;
            float sum30 = (float)_mm512_reduce_add_epi32(acc30) + (float)scalar30;
            float sum31 = (float)_mm512_reduce_add_epi32(acc31) + (float)scalar31;
            float sum32 = (float)_mm512_reduce_add_epi32(acc32) + (float)scalar32;
            float sum33 = (float)_mm512_reduce_add_epi32(acc33) + (float)scalar33;
            S[(i + 0) * n + j + 0] = sum00 * descale00 * scale;
            S[(i + 0) * n + j + 1] = sum01 * descale01 * scale;
            S[(i + 0) * n + j + 2] = sum02 * descale02 * scale;
            S[(i + 0) * n + j + 3] = sum03 * descale03 * scale;
            S[(i + 1) * n + j + 0] = sum10 * descale10 * scale;
            S[(i + 1) * n + j + 1] = sum11 * descale11 * scale;
            S[(i + 1) * n + j + 2] = sum12 * descale12 * scale;
            S[(i + 1) * n + j + 3] = sum13 * descale13 * scale;
            S[(i + 2) * n + j + 0] = sum20 * descale20 * scale;
            S[(i + 2) * n + j + 1] = sum21 * descale21 * scale;
            S[(i + 2) * n + j + 2] = sum22 * descale22 * scale;
            S[(i + 2) * n + j + 3] = sum23 * descale23 * scale;
            S[(i + 3) * n + j + 0] = sum30 * descale30 * scale;
            S[(i + 3) * n + j + 1] = sum31 * descale31 * scale;
            S[(i + 3) * n + j + 2] = sum32 * descale32 * scale;
            S[(i + 3) * n + j + 3] = sum33 * descale33 * scale;
        }
        for (; j + 1 < n; j += 2)
        {
            const signed char* k0 = K + j * d;
            const signed char* k1 = K + (j + 1) * d;

            __m512i acc00 = _mm512_setzero_epi32();
            __m512i acc01 = _mm512_setzero_epi32();
            __m512i acc10 = _mm512_setzero_epi32();
            __m512i acc11 = _mm512_setzero_epi32();
            __m512i acc20 = _mm512_setzero_epi32();
            __m512i acc21 = _mm512_setzero_epi32();
            __m512i acc30 = _mm512_setzero_epi32();
            __m512i acc31 = _mm512_setzero_epi32();
            int scalar00 = 0, scalar01 = 0, scalar10 = 0, scalar11 = 0;
            int scalar20 = 0, scalar21 = 0, scalar30 = 0, scalar31 = 0;

            for (int b = 0; b < num_blocks; b++)
            {
                int off = b * 32;
                int len = std::min(32, d - off);
                if (len <= 0) continue;
                if (len == 32)
                {
                    __m256i q0_32 = _mm256_loadu_si256((const __m256i*)(q0 + off));
                    __m256i q1_32 = _mm256_loadu_si256((const __m256i*)(q1 + off));
                    __m256i q2_32 = _mm256_loadu_si256((const __m256i*)(q2 + off));
                    __m256i q3_32 = _mm256_loadu_si256((const __m256i*)(q3 + off));
                    __m512i q0_512 = _mm512_cvtepi8_epi16(q0_32);
                    __m512i q1_512 = _mm512_cvtepi8_epi16(q1_32);
                    __m512i q2_512 = _mm512_cvtepi8_epi16(q2_32);
                    __m512i q3_512 = _mm512_cvtepi8_epi16(q3_32);

                    __m256i k0_32 = _mm256_loadu_si256((const __m256i*)(k0 + off));
                    __m256i k1_32 = _mm256_loadu_si256((const __m256i*)(k1 + off));
                    __m512i k0_512 = _mm512_cvtepi8_epi16(k0_32);
                    __m512i k1_512 = _mm512_cvtepi8_epi16(k1_32);

                    acc00 = _mm512_comp_dpwssd_epi32(acc00, q0_512, k0_512);
                    acc01 = _mm512_comp_dpwssd_epi32(acc01, q0_512, k1_512);
                    acc10 = _mm512_comp_dpwssd_epi32(acc10, q1_512, k0_512);
                    acc11 = _mm512_comp_dpwssd_epi32(acc11, q1_512, k1_512);
                    acc20 = _mm512_comp_dpwssd_epi32(acc20, q2_512, k0_512);
                    acc21 = _mm512_comp_dpwssd_epi32(acc21, q2_512, k1_512);
                    acc30 = _mm512_comp_dpwssd_epi32(acc30, q3_512, k0_512);
                    acc31 = _mm512_comp_dpwssd_epi32(acc31, q3_512, k1_512);
                }
                else
                {
                    int bs;
                    bs = qk_int8_dot_block_avx512_kernel(q0 + off, k0 + off, len);
                    scalar00 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q0 + off, k1 + off, len);
                    scalar01 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q1 + off, k0 + off, len);
                    scalar10 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q1 + off, k1 + off, len);
                    scalar11 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q2 + off, k0 + off, len);
                    scalar20 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q2 + off, k1 + off, len);
                    scalar21 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q3 + off, k0 + off, len);
                    scalar30 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q3 + off, k1 + off, len);
                    scalar31 += bs;
                }
            }
            float descale00 = qs0 * kscales[j];
            float descale01 = qs0 * kscales[j + 1];
            float descale10 = qs1 * kscales[j];
            float descale11 = qs1 * kscales[j + 1];
            float descale20 = qs2 * kscales[j];
            float descale21 = qs2 * kscales[j + 1];
            float descale30 = qs3 * kscales[j];
            float descale31 = qs3 * kscales[j + 1];
            float sum00 = (float)_mm512_reduce_add_epi32(acc00) + (float)scalar00;
            float sum01 = (float)_mm512_reduce_add_epi32(acc01) + (float)scalar01;
            float sum10 = (float)_mm512_reduce_add_epi32(acc10) + (float)scalar10;
            float sum11 = (float)_mm512_reduce_add_epi32(acc11) + (float)scalar11;
            float sum20 = (float)_mm512_reduce_add_epi32(acc20) + (float)scalar20;
            float sum21 = (float)_mm512_reduce_add_epi32(acc21) + (float)scalar21;
            float sum30 = (float)_mm512_reduce_add_epi32(acc30) + (float)scalar30;
            float sum31 = (float)_mm512_reduce_add_epi32(acc31) + (float)scalar31;
            S[(i + 0) * n + j] = sum00 * descale00 * scale;
            S[(i + 0) * n + j + 1] = sum01 * descale01 * scale;
            S[(i + 1) * n + j] = sum10 * descale10 * scale;
            S[(i + 1) * n + j + 1] = sum11 * descale11 * scale;
            S[(i + 2) * n + j] = sum20 * descale20 * scale;
            S[(i + 2) * n + j + 1] = sum21 * descale21 * scale;
            S[(i + 3) * n + j] = sum30 * descale30 * scale;
            S[(i + 3) * n + j + 1] = sum31 * descale31 * scale;
        }
        for (; j < n; j++)
        {
            const signed char* kptr = K + j * d;

            __m512i acc0 = _mm512_setzero_epi32();
            __m512i acc1 = _mm512_setzero_epi32();
            __m512i acc2 = _mm512_setzero_epi32();
            __m512i acc3 = _mm512_setzero_epi32();
            int scalar0 = 0, scalar1 = 0, scalar2 = 0, scalar3 = 0;

            for (int b = 0; b < num_blocks; b++)
            {
                int off = b * 32;
                int len = std::min(32, d - off);
                if (len <= 0) continue;
                if (len == 32)
                {
                    __m256i q0_32 = _mm256_loadu_si256((const __m256i*)(q0 + off));
                    __m256i q1_32 = _mm256_loadu_si256((const __m256i*)(q1 + off));
                    __m256i q2_32 = _mm256_loadu_si256((const __m256i*)(q2 + off));
                    __m256i q3_32 = _mm256_loadu_si256((const __m256i*)(q3 + off));
                    __m512i q0_512 = _mm512_cvtepi8_epi16(q0_32);
                    __m512i q1_512 = _mm512_cvtepi8_epi16(q1_32);
                    __m512i q2_512 = _mm512_cvtepi8_epi16(q2_32);
                    __m512i q3_512 = _mm512_cvtepi8_epi16(q3_32);

                    __m256i k_32 = _mm256_loadu_si256((const __m256i*)(kptr + off));
                    __m512i k_512 = _mm512_cvtepi8_epi16(k_32);

                    acc0 = _mm512_comp_dpwssd_epi32(acc0, q0_512, k_512);
                    acc1 = _mm512_comp_dpwssd_epi32(acc1, q1_512, k_512);
                    acc2 = _mm512_comp_dpwssd_epi32(acc2, q2_512, k_512);
                    acc3 = _mm512_comp_dpwssd_epi32(acc3, q3_512, k_512);
                }
                else
                {
                    int bs;
                    bs = qk_int8_dot_block_avx512_kernel(q0 + off, kptr + off, len);
                    scalar0 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q1 + off, kptr + off, len);
                    scalar1 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q2 + off, kptr + off, len);
                    scalar2 += bs;
                    bs = qk_int8_dot_block_avx512_kernel(q3 + off, kptr + off, len);
                    scalar3 += bs;
                }
            }
            float descale = kscales[j];
            float sum0 = (float)_mm512_reduce_add_epi32(acc0) + (float)scalar0;
            float sum1 = (float)_mm512_reduce_add_epi32(acc1) + (float)scalar1;
            float sum2 = (float)_mm512_reduce_add_epi32(acc2) + (float)scalar2;
            float sum3 = (float)_mm512_reduce_add_epi32(acc3) + (float)scalar3;
            S[(i + 0) * n + j] = sum0 * qs0 * descale * scale;
            S[(i + 1) * n + j] = sum1 * qs1 * descale * scale;
            S[(i + 2) * n + j] = sum2 * qs2 * descale * scale;
            S[(i + 3) * n + j] = sum3 * qs3 * descale * scale;
        }
    }
    for (; i < m; i++)
    {
        qk_int8_gemm_row_avx512_kernel(S + i * n, Q + i * d, K, qscales[i], kscales, n, d, scale);
    }
}
#endif // __AVX512F__

#if __AVX512VNNI__
static inline void qk_int8_gemm_tiled_avx512vnni_kernel(float* S,
        const signed char* Q, const signed char* K,
        const float* qscales, const float* kscales,
        int m, int n, int d, float scale)
{
    const int num_blocks_64 = d / 64;
    int i = 0;
    for (; i + 4 <= m; i += 4)
    {
        const signed char* q0 = Q + (i + 0) * d;
        const signed char* q1 = Q + (i + 1) * d;
        const signed char* q2 = Q + (i + 2) * d;
        const signed char* q3 = Q + (i + 3) * d;
        float qs0 = qscales[i + 0];
        float qs1 = qscales[i + 1];
        float qs2 = qscales[i + 2];
        float qs3 = qscales[i + 3];

        int qsum0_64byte = 0, qsum1_64byte = 0, qsum2_64byte = 0, qsum3_64byte = 0;
        if (num_blocks_64 > 0)
        {
            __m512i ones = _mm512_set1_epi8(1);
            __m512i sum0 = _mm512_setzero_epi32();
            __m512i sum1 = _mm512_setzero_epi32();
            __m512i sum2 = _mm512_setzero_epi32();
            __m512i sum3 = _mm512_setzero_epi32();
            for (int b = 0; b < num_blocks_64; b++)
            {
                int off = b * 64;
                sum0 = _mm512_dpbusd_epi32(sum0, ones, _mm512_loadu_si512((const __m512i*)(q0 + off)));
                sum1 = _mm512_dpbusd_epi32(sum1, ones, _mm512_loadu_si512((const __m512i*)(q1 + off)));
                sum2 = _mm512_dpbusd_epi32(sum2, ones, _mm512_loadu_si512((const __m512i*)(q2 + off)));
                sum3 = _mm512_dpbusd_epi32(sum3, ones, _mm512_loadu_si512((const __m512i*)(q3 + off)));
            }
            qsum0_64byte = _mm512_reduce_add_epi32(sum0);
            qsum1_64byte = _mm512_reduce_add_epi32(sum1);
            qsum2_64byte = _mm512_reduce_add_epi32(sum2);
            qsum3_64byte = _mm512_reduce_add_epi32(sum3);
        }

        int j = 0;
        for (; j + 3 < n; j += 4)
        {
            const signed char* k0 = K + j * d;
            const signed char* k1 = K + (j + 1) * d;
            const signed char* k2 = K + (j + 2) * d;
            const signed char* k3 = K + (j + 3) * d;

            __m512i acc00 = _mm512_setzero_epi32();
            __m512i acc01 = _mm512_setzero_epi32();
            __m512i acc02 = _mm512_setzero_epi32();
            __m512i acc03 = _mm512_setzero_epi32();
            __m512i acc10 = _mm512_setzero_epi32();
            __m512i acc11 = _mm512_setzero_epi32();
            __m512i acc12 = _mm512_setzero_epi32();
            __m512i acc13 = _mm512_setzero_epi32();
            __m512i acc20 = _mm512_setzero_epi32();
            __m512i acc21 = _mm512_setzero_epi32();
            __m512i acc22 = _mm512_setzero_epi32();
            __m512i acc23 = _mm512_setzero_epi32();
            __m512i acc30 = _mm512_setzero_epi32();
            __m512i acc31 = _mm512_setzero_epi32();
            __m512i acc32 = _mm512_setzero_epi32();
            __m512i acc33 = _mm512_setzero_epi32();
            int scalar00 = 0, scalar01 = 0, scalar02 = 0, scalar03 = 0;
            int scalar10 = 0, scalar11 = 0, scalar12 = 0, scalar13 = 0;
            int scalar20 = 0, scalar21 = 0, scalar22 = 0, scalar23 = 0;
            int scalar30 = 0, scalar31 = 0, scalar32 = 0, scalar33 = 0;

            for (int b = 0; b < num_blocks_64; b++)
            {
                int off = b * 64;
                __m512i q0_512 = _mm512_loadu_si512((const __m512i*)(q0 + off));
                __m512i q1_512 = _mm512_loadu_si512((const __m512i*)(q1 + off));
                __m512i q2_512 = _mm512_loadu_si512((const __m512i*)(q2 + off));
                __m512i q3_512 = _mm512_loadu_si512((const __m512i*)(q3 + off));

                __m512i k0_512 = _mm512_loadu_si512((const __m512i*)(k0 + off));
                __m512i k1_512 = _mm512_loadu_si512((const __m512i*)(k1 + off));
                __m512i k2_512 = _mm512_loadu_si512((const __m512i*)(k2 + off));
                __m512i k3_512 = _mm512_loadu_si512((const __m512i*)(k3 + off));
                __m512i k0_u8 = _mm512_add_epi8(k0_512, _mm512_set1_epi8(128));
                __m512i k1_u8 = _mm512_add_epi8(k1_512, _mm512_set1_epi8(128));
                __m512i k2_u8 = _mm512_add_epi8(k2_512, _mm512_set1_epi8(128));
                __m512i k3_u8 = _mm512_add_epi8(k3_512, _mm512_set1_epi8(128));

                acc00 = _mm512_dpbusd_epi32(acc00, k0_u8, q0_512);
                acc01 = _mm512_dpbusd_epi32(acc01, k1_u8, q0_512);
                acc02 = _mm512_dpbusd_epi32(acc02, k2_u8, q0_512);
                acc03 = _mm512_dpbusd_epi32(acc03, k3_u8, q0_512);
                acc10 = _mm512_dpbusd_epi32(acc10, k0_u8, q1_512);
                acc11 = _mm512_dpbusd_epi32(acc11, k1_u8, q1_512);
                acc12 = _mm512_dpbusd_epi32(acc12, k2_u8, q1_512);
                acc13 = _mm512_dpbusd_epi32(acc13, k3_u8, q1_512);
                acc20 = _mm512_dpbusd_epi32(acc20, k0_u8, q2_512);
                acc21 = _mm512_dpbusd_epi32(acc21, k1_u8, q2_512);
                acc22 = _mm512_dpbusd_epi32(acc22, k2_u8, q2_512);
                acc23 = _mm512_dpbusd_epi32(acc23, k3_u8, q2_512);
                acc30 = _mm512_dpbusd_epi32(acc30, k0_u8, q3_512);
                acc31 = _mm512_dpbusd_epi32(acc31, k1_u8, q3_512);
                acc32 = _mm512_dpbusd_epi32(acc32, k2_u8, q3_512);
                acc33 = _mm512_dpbusd_epi32(acc33, k3_u8, q3_512);
            }

            int tail_start = num_blocks_64 * 64;
            if (tail_start < d)
            {
                for (int k = tail_start; k < d; k++)
                {
                    scalar00 += q0[k] * k0[k];
                    scalar01 += q0[k] * k1[k];
                    scalar02 += q0[k] * k2[k];
                    scalar03 += q0[k] * k3[k];
                    scalar10 += q1[k] * k0[k];
                    scalar11 += q1[k] * k1[k];
                    scalar12 += q1[k] * k2[k];
                    scalar13 += q1[k] * k3[k];
                    scalar20 += q2[k] * k0[k];
                    scalar21 += q2[k] * k1[k];
                    scalar22 += q2[k] * k2[k];
                    scalar23 += q2[k] * k3[k];
                    scalar30 += q3[k] * k0[k];
                    scalar31 += q3[k] * k1[k];
                    scalar32 += q3[k] * k2[k];
                    scalar33 += q3[k] * k3[k];
                }
            }

            float descale00 = qs0 * kscales[j];
            float descale01 = qs0 * kscales[j + 1];
            float descale02 = qs0 * kscales[j + 2];
            float descale03 = qs0 * kscales[j + 3];
            float descale10 = qs1 * kscales[j];
            float descale11 = qs1 * kscales[j + 1];
            float descale12 = qs1 * kscales[j + 2];
            float descale13 = qs1 * kscales[j + 3];
            float descale20 = qs2 * kscales[j];
            float descale21 = qs2 * kscales[j + 1];
            float descale22 = qs2 * kscales[j + 2];
            float descale23 = qs2 * kscales[j + 3];
            float descale30 = qs3 * kscales[j];
            float descale31 = qs3 * kscales[j + 1];
            float descale32 = qs3 * kscales[j + 2];
            float descale33 = qs3 * kscales[j + 3];
            float sum00 = (float)(_mm512_reduce_add_epi32(acc00) - 128 * qsum0_64byte) + (float)scalar00;
            float sum01 = (float)(_mm512_reduce_add_epi32(acc01) - 128 * qsum0_64byte) + (float)scalar01;
            float sum02 = (float)(_mm512_reduce_add_epi32(acc02) - 128 * qsum0_64byte) + (float)scalar02;
            float sum03 = (float)(_mm512_reduce_add_epi32(acc03) - 128 * qsum0_64byte) + (float)scalar03;
            float sum10 = (float)(_mm512_reduce_add_epi32(acc10) - 128 * qsum1_64byte) + (float)scalar10;
            float sum11 = (float)(_mm512_reduce_add_epi32(acc11) - 128 * qsum1_64byte) + (float)scalar11;
            float sum12 = (float)(_mm512_reduce_add_epi32(acc12) - 128 * qsum1_64byte) + (float)scalar12;
            float sum13 = (float)(_mm512_reduce_add_epi32(acc13) - 128 * qsum1_64byte) + (float)scalar13;
            float sum20 = (float)(_mm512_reduce_add_epi32(acc20) - 128 * qsum2_64byte) + (float)scalar20;
            float sum21 = (float)(_mm512_reduce_add_epi32(acc21) - 128 * qsum2_64byte) + (float)scalar21;
            float sum22 = (float)(_mm512_reduce_add_epi32(acc22) - 128 * qsum2_64byte) + (float)scalar22;
            float sum23 = (float)(_mm512_reduce_add_epi32(acc23) - 128 * qsum2_64byte) + (float)scalar23;
            float sum30 = (float)(_mm512_reduce_add_epi32(acc30) - 128 * qsum3_64byte) + (float)scalar30;
            float sum31 = (float)(_mm512_reduce_add_epi32(acc31) - 128 * qsum3_64byte) + (float)scalar31;
            float sum32 = (float)(_mm512_reduce_add_epi32(acc32) - 128 * qsum3_64byte) + (float)scalar32;
            float sum33 = (float)(_mm512_reduce_add_epi32(acc33) - 128 * qsum3_64byte) + (float)scalar33;
            S[(i + 0) * n + j + 0] = sum00 * descale00 * scale;
            S[(i + 0) * n + j + 1] = sum01 * descale01 * scale;
            S[(i + 0) * n + j + 2] = sum02 * descale02 * scale;
            S[(i + 0) * n + j + 3] = sum03 * descale03 * scale;
            S[(i + 1) * n + j + 0] = sum10 * descale10 * scale;
            S[(i + 1) * n + j + 1] = sum11 * descale11 * scale;
            S[(i + 1) * n + j + 2] = sum12 * descale12 * scale;
            S[(i + 1) * n + j + 3] = sum13 * descale13 * scale;
            S[(i + 2) * n + j + 0] = sum20 * descale20 * scale;
            S[(i + 2) * n + j + 1] = sum21 * descale21 * scale;
            S[(i + 2) * n + j + 2] = sum22 * descale22 * scale;
            S[(i + 2) * n + j + 3] = sum23 * descale23 * scale;
            S[(i + 3) * n + j + 0] = sum30 * descale30 * scale;
            S[(i + 3) * n + j + 1] = sum31 * descale31 * scale;
            S[(i + 3) * n + j + 2] = sum32 * descale32 * scale;
            S[(i + 3) * n + j + 3] = sum33 * descale33 * scale;
        }
        for (; j + 1 < n; j += 2)
        {
            const signed char* k0 = K + j * d;
            const signed char* k1 = K + (j + 1) * d;

            __m512i acc00 = _mm512_setzero_epi32();
            __m512i acc01 = _mm512_setzero_epi32();
            __m512i acc10 = _mm512_setzero_epi32();
            __m512i acc11 = _mm512_setzero_epi32();
            __m512i acc20 = _mm512_setzero_epi32();
            __m512i acc21 = _mm512_setzero_epi32();
            __m512i acc30 = _mm512_setzero_epi32();
            __m512i acc31 = _mm512_setzero_epi32();
            int scalar00 = 0, scalar01 = 0, scalar10 = 0, scalar11 = 0;
            int scalar20 = 0, scalar21 = 0, scalar30 = 0, scalar31 = 0;

            for (int b = 0; b < num_blocks_64; b++)
            {
                int off = b * 64;
                __m512i q0_512 = _mm512_loadu_si512((const __m512i*)(q0 + off));
                __m512i q1_512 = _mm512_loadu_si512((const __m512i*)(q1 + off));
                __m512i q2_512 = _mm512_loadu_si512((const __m512i*)(q2 + off));
                __m512i q3_512 = _mm512_loadu_si512((const __m512i*)(q3 + off));

                __m512i k0_512 = _mm512_loadu_si512((const __m512i*)(k0 + off));
                __m512i k1_512 = _mm512_loadu_si512((const __m512i*)(k1 + off));
                __m512i k0_u8 = _mm512_add_epi8(k0_512, _mm512_set1_epi8(128));
                __m512i k1_u8 = _mm512_add_epi8(k1_512, _mm512_set1_epi8(128));

                acc00 = _mm512_dpbusd_epi32(acc00, k0_u8, q0_512);
                acc01 = _mm512_dpbusd_epi32(acc01, k1_u8, q0_512);
                acc10 = _mm512_dpbusd_epi32(acc10, k0_u8, q1_512);
                acc11 = _mm512_dpbusd_epi32(acc11, k1_u8, q1_512);
                acc20 = _mm512_dpbusd_epi32(acc20, k0_u8, q2_512);
                acc21 = _mm512_dpbusd_epi32(acc21, k1_u8, q2_512);
                acc30 = _mm512_dpbusd_epi32(acc30, k0_u8, q3_512);
                acc31 = _mm512_dpbusd_epi32(acc31, k1_u8, q3_512);
            }

            int tail_start = num_blocks_64 * 64;
            if (tail_start < d)
            {
                for (int k = tail_start; k < d; k++)
                {
                    scalar00 += q0[k] * k0[k];
                    scalar01 += q0[k] * k1[k];
                    scalar10 += q1[k] * k0[k];
                    scalar11 += q1[k] * k1[k];
                    scalar20 += q2[k] * k0[k];
                    scalar21 += q2[k] * k1[k];
                    scalar30 += q3[k] * k0[k];
                    scalar31 += q3[k] * k1[k];
                }
            }

            float descale00 = qs0 * kscales[j];
            float descale01 = qs0 * kscales[j + 1];
            float descale10 = qs1 * kscales[j];
            float descale11 = qs1 * kscales[j + 1];
            float descale20 = qs2 * kscales[j];
            float descale21 = qs2 * kscales[j + 1];
            float descale30 = qs3 * kscales[j];
            float descale31 = qs3 * kscales[j + 1];
            float sum00 = (float)(_mm512_reduce_add_epi32(acc00) - 128 * qsum0_64byte) + (float)scalar00;
            float sum01 = (float)(_mm512_reduce_add_epi32(acc01) - 128 * qsum0_64byte) + (float)scalar01;
            float sum10 = (float)(_mm512_reduce_add_epi32(acc10) - 128 * qsum1_64byte) + (float)scalar10;
            float sum11 = (float)(_mm512_reduce_add_epi32(acc11) - 128 * qsum1_64byte) + (float)scalar11;
            float sum20 = (float)(_mm512_reduce_add_epi32(acc20) - 128 * qsum2_64byte) + (float)scalar20;
            float sum21 = (float)(_mm512_reduce_add_epi32(acc21) - 128 * qsum2_64byte) + (float)scalar21;
            float sum30 = (float)(_mm512_reduce_add_epi32(acc30) - 128 * qsum3_64byte) + (float)scalar30;
            float sum31 = (float)(_mm512_reduce_add_epi32(acc31) - 128 * qsum3_64byte) + (float)scalar31;
            S[(i + 0) * n + j] = sum00 * descale00 * scale;
            S[(i + 0) * n + j + 1] = sum01 * descale01 * scale;
            S[(i + 1) * n + j] = sum10 * descale10 * scale;
            S[(i + 1) * n + j + 1] = sum11 * descale11 * scale;
            S[(i + 2) * n + j] = sum20 * descale20 * scale;
            S[(i + 2) * n + j + 1] = sum21 * descale21 * scale;
            S[(i + 3) * n + j] = sum30 * descale30 * scale;
            S[(i + 3) * n + j + 1] = sum31 * descale31 * scale;
        }
        for (; j < n; j++)
        {
            const signed char* kptr = K + j * d;

            __m512i acc0 = _mm512_setzero_epi32();
            __m512i acc1 = _mm512_setzero_epi32();
            __m512i acc2 = _mm512_setzero_epi32();
            __m512i acc3 = _mm512_setzero_epi32();
            int scalar0 = 0, scalar1 = 0, scalar2 = 0, scalar3 = 0;

            for (int b = 0; b < num_blocks_64; b++)
            {
                int off = b * 64;
                __m512i q0_512 = _mm512_loadu_si512((const __m512i*)(q0 + off));
                __m512i q1_512 = _mm512_loadu_si512((const __m512i*)(q1 + off));
                __m512i q2_512 = _mm512_loadu_si512((const __m512i*)(q2 + off));
                __m512i q3_512 = _mm512_loadu_si512((const __m512i*)(q3 + off));

                __m512i k_512 = _mm512_loadu_si512((const __m512i*)(kptr + off));
                __m512i k_u8 = _mm512_add_epi8(k_512, _mm512_set1_epi8(128));

                acc0 = _mm512_dpbusd_epi32(acc0, k_u8, q0_512);
                acc1 = _mm512_dpbusd_epi32(acc1, k_u8, q1_512);
                acc2 = _mm512_dpbusd_epi32(acc2, k_u8, q2_512);
                acc3 = _mm512_dpbusd_epi32(acc3, k_u8, q3_512);
            }

            int tail_start = num_blocks_64 * 64;
            if (tail_start < d)
            {
                for (int k = tail_start; k < d; k++)
                {
                    scalar0 += q0[k] * kptr[k];
                    scalar1 += q1[k] * kptr[k];
                    scalar2 += q2[k] * kptr[k];
                    scalar3 += q3[k] * kptr[k];
                }
            }

            float descale = kscales[j];
            float sum0 = (float)(_mm512_reduce_add_epi32(acc0) - 128 * qsum0_64byte) + (float)scalar0;
            float sum1 = (float)(_mm512_reduce_add_epi32(acc1) - 128 * qsum1_64byte) + (float)scalar1;
            float sum2 = (float)(_mm512_reduce_add_epi32(acc2) - 128 * qsum2_64byte) + (float)scalar2;
            float sum3 = (float)(_mm512_reduce_add_epi32(acc3) - 128 * qsum3_64byte) + (float)scalar3;
            S[(i + 0) * n + j] = sum0 * qs0 * descale * scale;
            S[(i + 1) * n + j] = sum1 * qs1 * descale * scale;
            S[(i + 2) * n + j] = sum2 * qs2 * descale * scale;
            S[(i + 3) * n + j] = sum3 * qs3 * descale * scale;
        }
    }
    for (; i < m; i++)
    {
        qk_int8_gemm_row_avx512vnni_kernel(S + i * n, Q + i * d, K, qscales[i], kscales, n, d, scale);
    }
}
#endif // __AVX512VNNI__

static void qk_int8_gemm_tiled(float* S,
                               const signed char* Q, const signed char* K,
                               const float* qscales, const float* kscales,
                               int m, int n, int d, float scale)
{
#if __AVX512F__
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        qk_int8_gemm_tiled_avx512vnni(S, Q, K, qscales, kscales, m, n, d, scale);
        return;
    }
#endif
#if __AVX512VNNI__
    qk_int8_gemm_tiled_avx512vnni_kernel(S, Q, K, qscales, kscales, m, n, d, scale);
#else
    qk_int8_gemm_tiled_avx512_kernel(S, Q, K, qscales, kscales, m, n, d, scale);
#endif
#else
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        qk_int8_gemm_tiled_avx512vnni(S, Q, K, qscales, kscales, m, n, d, scale);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_AVXVNNIINT8 && __AVX__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni_int8())
    {
        qk_int8_gemm_tiled_avxvnniint8(S, Q, K, qscales, kscales, m, n, d, scale);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        qk_int8_gemm_tiled_avxvnni(S, Q, K, qscales, kscales, m, n, d, scale);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVXVNNIINT8__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        qk_int8_gemm_tiled_avx2(S, Q, K, qscales, kscales, m, n, d, scale);
        return;
    }
#endif
#if __AVX2__
    qk_int8_gemm_tiled_avx2_kernel(S, Q, K, qscales, kscales, m, n, d, scale);
#elif __SSE2__
    qk_int8_gemm_tiled_sse2_kernel(S, Q, K, qscales, kscales, m, n, d, scale);
#else
    qk_int8_gemm_tiled_scalar_kernel(S, Q, K, qscales, kscales, m, n, d, scale);
#endif
#endif
}

// ------------------- Decode PV GEMV Int8 -------------------

static inline void decode_pv_gemv_int8_scalar_kernel(float* out, const float* s,
        const signed char* V, const float* vscales,
        int n_start, int block_n, int out_d)
{
    const int num_blocks = (out_d + 31) / 32;
    for (int vb = 0; vb < num_blocks; vb++)
    {
        int k_start = vb * 32;
        int k_end = k_start + 32 < out_d ? k_start + 32 : out_d;
        for (int k = k_start; k < k_end; k++)
            out[k] = 0.f;

        for (int j = 0; j < block_n; j++)
        {
            float p = s[j];
            float inv_scale = 1.f / vscales[(n_start + j) * num_blocks + vb];
            const signed char* vptr = V + (n_start + j) * out_d + k_start;
            for (int k = k_start; k < k_end; k++)
                out[k] += p * (float)vptr[k - k_start] * inv_scale;
        }
    }
}

#if __SSE2__
static inline void decode_pv_gemv_int8_sse2_kernel(float* out, const float* s,
        const signed char* V, const float* vscales,
        int n_start, int block_n, int out_d)
{
    const int num_blocks = (out_d + 31) / 32;
    for (int vb = 0; vb < num_blocks; vb++)
    {
        int k_start = vb * 32;
        int k_end = k_start + 32 < out_d ? k_start + 32 : out_d;

        int k = k_start;
        for (; k + 3 < k_end; k += 4)
        {
            __m128 oval = _mm_setzero_ps();
            for (int j = 0; j < block_n; j++)
            {
                float p_invscale = s[j] / vscales[(n_start + j) * num_blocks + vb];
                __m128 pvec = _mm_set1_ps(p_invscale);
                __m128i v8 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          V[(n_start + j) * out_d + k + 3], V[(n_start + j) * out_d + k + 2],
                                          V[(n_start + j) * out_d + k + 1], V[(n_start + j) * out_d + k + 0]);
                __m128i v32 = _mm_cvtepi8_epi32(v8);
                __m128 vfp = _mm_cvtepi32_ps(v32);
                oval = _mm_add_ps(oval, _mm_mul_ps(pvec, vfp));
            }
            _mm_storeu_ps(out + k, _mm_add_ps(_mm_loadu_ps(out + k), oval));
        }
        for (; k < k_end; k++)
        {
            float sum = 0.f;
            for (int j = 0; j < block_n; j++)
            {
                float p_invscale = s[j] / vscales[(n_start + j) * num_blocks + vb];
                sum += p_invscale * V[(n_start + j) * out_d + k];
            }
            out[k] += sum;
        }
    }
}
#endif // __SSE2__

#if __AVX2__
inline void decode_pv_gemv_int8_avx2_kernel(float* out, const float* s,
        const signed char* V, const float* vscales,
        int n_start, int block_n, int out_d)
{
    const int num_blocks = (out_d + 31) / 32;
    for (int vb = 0; vb < num_blocks; vb++)
    {
        int k_start = vb * 32;
        int k_end = k_start + 32 < out_d ? k_start + 32 : out_d;

        int k = k_start;
        for (; k + 7 < k_end; k += 8)
        {
            __m256 oval = _mm256_setzero_ps();
            for (int j = 0; j < block_n; j++)
            {
                float p_invscale = s[j] / vscales[(n_start + j) * num_blocks + vb];
                __m256 pvec = _mm256_set1_ps(p_invscale);
                __m128i v8 = _mm_loadl_epi64((const __m128i*)(V + (n_start + j) * out_d + k));
                __m256i v32 = _mm256_cvtepi8_epi32(v8);
                __m256 vfp = _mm256_cvtepi32_ps(v32);
                oval = _mm256_fmadd_ps(pvec, vfp, oval);
            }
            _mm256_storeu_ps(out + k, _mm256_add_ps(_mm256_loadu_ps(out + k), oval));
        }
        for (; k + 3 < k_end; k += 4)
        {
            __m128 oval = _mm_setzero_ps();
            for (int j = 0; j < block_n; j++)
            {
                float p_invscale = s[j] / vscales[(n_start + j) * num_blocks + vb];
                __m128 pvec = _mm_set1_ps(p_invscale);
                __m128i v8 = _mm_cvtsi32_si128(*(const int*)(V + (n_start + j) * out_d + k));
                __m128i v32 = _mm_cvtepi8_epi32(v8);
                __m128 vfp = _mm_cvtepi32_ps(v32);
                oval = _mm_add_ps(oval, _mm_mul_ps(pvec, vfp));
            }
            _mm_storeu_ps(out + k, oval);
        }
        for (; k < k_end; k++)
        {
            float sum = 0.f;
            for (int j = 0; j < block_n; j++)
                sum += s[j] / vscales[(n_start + j) * num_blocks + vb] * V[(n_start + j) * out_d + k];
            out[k] = sum;
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static inline void decode_pv_gemv_int8_avx512_kernel(float* out, const float* s,
        const signed char* V, const float* vscales,
        int n_start, int block_n, int out_d)
{
    const int num_blocks = (out_d + 31) / 32;
    int j = 0;
    for (; j + 1 < block_n; j += 2)
    {
        float p0 = s[j];
        float p1 = s[j + 1];
        const signed char* v0 = V + (n_start + j) * out_d;
        const signed char* v1 = V + (n_start + j + 1) * out_d;

        int k = 0;
        for (; k + 31 < out_d; k += 32)
        {
            int vb = k / 32;
            float p0_invscale = p0 / vscales[(n_start + j) * num_blocks + vb];
            float p1_invscale = p1 / vscales[(n_start + j + 1) * num_blocks + vb];

            __m512 oval0 = _mm512_loadu_ps(out + k);
            __m512 oval1 = _mm512_loadu_ps(out + k + 16);

            __m128i v8_0a = _mm_loadu_si128((const __m128i*)(v0 + k));
            __m128i v8_0b = _mm_loadu_si128((const __m128i*)(v0 + k + 16));
            oval0 = _mm512_fmadd_ps(_mm512_set1_ps(p0_invscale), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v8_0a)), oval0);
            oval1 = _mm512_fmadd_ps(_mm512_set1_ps(p0_invscale), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v8_0b)), oval1);

            __m128i v8_1a = _mm_loadu_si128((const __m128i*)(v1 + k));
            __m128i v8_1b = _mm_loadu_si128((const __m128i*)(v1 + k + 16));
            oval0 = _mm512_fmadd_ps(_mm512_set1_ps(p1_invscale), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v8_1a)), oval0);
            oval1 = _mm512_fmadd_ps(_mm512_set1_ps(p1_invscale), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v8_1b)), oval1);

            _mm512_storeu_ps(out + k, oval0);
            _mm512_storeu_ps(out + k + 16, oval1);
        }
        for (; k + 15 < out_d; k += 16)
        {
            int vb = k / 32;
            float p0_invscale = p0 / vscales[(n_start + j) * num_blocks + vb];
            float p1_invscale = p1 / vscales[(n_start + j + 1) * num_blocks + vb];

            __m512 oval = _mm512_loadu_ps(out + k);

            __m128i v8_0 = _mm_loadu_si128((const __m128i*)(v0 + k));
            oval = _mm512_fmadd_ps(_mm512_set1_ps(p0_invscale), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v8_0)), oval);

            __m128i v8_1 = _mm_loadu_si128((const __m128i*)(v1 + k));
            oval = _mm512_fmadd_ps(_mm512_set1_ps(p1_invscale), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v8_1)), oval);

            _mm512_storeu_ps(out + k, oval);
        }
        for (; k < out_d; k++)
        {
            int vb = k / 32;
            float p0_invscale = p0 / vscales[(n_start + j) * num_blocks + vb];
            float p1_invscale = p1 / vscales[(n_start + j + 1) * num_blocks + vb];
            out[k] += p0_invscale * v0[k] + p1_invscale * v1[k];
        }
    }
    for (; j < block_n; j++)
    {
        float p = s[j];
        const signed char* v = V + (n_start + j) * out_d;

        int k = 0;
        for (; k + 31 < out_d; k += 32)
        {
            int vb = k / 32;
            float p_invscale = p / vscales[(n_start + j) * num_blocks + vb];

            __m512 oval0 = _mm512_loadu_ps(out + k);
            __m512 oval1 = _mm512_loadu_ps(out + k + 16);

            __m128i v8_0 = _mm_loadu_si128((const __m128i*)(v + k));
            __m128i v8_1 = _mm_loadu_si128((const __m128i*)(v + k + 16));
            oval0 = _mm512_fmadd_ps(_mm512_set1_ps(p_invscale), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v8_0)), oval0);
            oval1 = _mm512_fmadd_ps(_mm512_set1_ps(p_invscale), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v8_1)), oval1);

            _mm512_storeu_ps(out + k, oval0);
            _mm512_storeu_ps(out + k + 16, oval1);
        }
        for (; k + 15 < out_d; k += 16)
        {
            int vb = k / 32;
            float p_invscale = p / vscales[(n_start + j) * num_blocks + vb];

            __m512 oval = _mm512_loadu_ps(out + k);
            __m128i v8 = _mm_loadu_si128((const __m128i*)(v + k));
            oval = _mm512_fmadd_ps(_mm512_set1_ps(p_invscale), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v8)), oval);
            _mm512_storeu_ps(out + k, oval);
        }
        for (; k < out_d; k++)
        {
            int vb = k / 32;
            float p_invscale = p / vscales[(n_start + j) * num_blocks + vb];
            out[k] += p_invscale * v[k];
        }
    }
}
#endif // __AVX512F__

static void decode_pv_gemv_int8(float* out, const float* s,
                                const signed char* V, const float* vscales,
                                int n_start, int block_n, int out_d)
{
#if __AVX512F__
    decode_pv_gemv_int8_avx512_kernel(out, s, V, vscales, n_start, block_n, out_d);
#else
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        decode_pv_gemv_int8_avx2(out, s, V, vscales, n_start, block_n, out_d);
        return;
    }
#endif
#if __AVX2__
    decode_pv_gemv_int8_avx2(out, s, V, vscales, n_start, block_n, out_d);
#elif __SSE2__
    decode_pv_gemv_int8_sse2_kernel(out, s, V, vscales, n_start, block_n, out_d);
#else
    decode_pv_gemv_int8_scalar_kernel(out, s, V, vscales, n_start, block_n, out_d);
#endif
#endif
}

// ------------------- Prefill PV Float×Int8 GEMM Row-wise -------------------

static inline void pv_float_int8_gemm_row_scalar_kernel(float* out, const float* p_row,
        const signed char* V, const float* vscales,
        int n, int out_d)
{
    const int num_blocks = (out_d + 31) / 32;
    for (int vb = 0; vb < num_blocks; vb++)
    {
        int k_start = vb * 32;
        int k_end = k_start + 32 < out_d ? k_start + 32 : out_d;
        for (int k = k_start; k < k_end; k++)
            out[k] = 0.f;

        for (int j = 0; j < n; j++)
        {
            float p = p_row[j];
            float inv_scale = 1.f / vscales[j * num_blocks + vb];
            const signed char* vptr = V + j * out_d + k_start;
            for (int k = k_start; k < k_end; k++)
                out[k] += p * (float)vptr[k - k_start] * inv_scale;
        }
    }
}

#if __SSE2__
static inline void pv_float_int8_gemm_row_sse2_kernel(float* out, const float* p_row,
        const signed char* V, const float* vscales,
        int n, int out_d)
{
    const int num_blocks = (out_d + 31) / 32;
    for (int vb = 0; vb < num_blocks; vb++)
    {
        int k_start = vb * 32;
        int k_end = k_start + 32 < out_d ? k_start + 32 : out_d;
        int k = k_start;
        for (; k + 3 < k_end; k += 4)
        {
            __m128 oval = _mm_setzero_ps();
            for (int j = 0; j < n; j++)
            {
                float p_invscale = p_row[j] / vscales[j * num_blocks + vb];
                __m128 pvec = _mm_set1_ps(p_invscale);
                __m128i v8 = _mm_cvtsi32_si128(*(const int*)(V + j * out_d + k));
                __m128i v32 = _mm_cvtepi8_epi32(v8);
                __m128 vfp = _mm_cvtepi32_ps(v32);
                oval = _mm_add_ps(oval, _mm_mul_ps(pvec, vfp));
            }
            _mm_storeu_ps(out + k, oval);
        }
        for (; k < k_end; k++)
        {
            float sum = 0.f;
            for (int j = 0; j < n; j++)
                sum += p_row[j] / vscales[j * num_blocks + vb] * V[j * out_d + k];
            out[k] = sum;
        }
    }
}
#endif // __SSE2__

#if __AVX2__
inline void pv_float_int8_gemm_row_avx2_kernel(float* out, const float* p_row,
        const signed char* V, const float* vscales,
        int n, int out_d)
{
    const int num_blocks = (out_d + 31) / 32;
    for (int vb = 0; vb < num_blocks; vb++)
    {
        int k_start = vb * 32;
        int k_end = k_start + 32 < out_d ? k_start + 32 : out_d;
        int k = k_start;
        for (; k + 7 < k_end; k += 8)
        {
            __m256 oval = _mm256_setzero_ps();
            for (int j = 0; j < n; j++)
            {
                float p_invscale = p_row[j] / vscales[j * num_blocks + vb];
                __m256 pvec = _mm256_set1_ps(p_invscale);
                __m128i v8 = _mm_loadl_epi64((const __m128i*)(V + j * out_d + k));
                __m256i v32 = _mm256_cvtepi8_epi32(v8);
                __m256 vfp = _mm256_cvtepi32_ps(v32);
                oval = _mm256_fmadd_ps(pvec, vfp, oval);
            }
            _mm256_storeu_ps(out + k, oval);
        }
        for (; k + 3 < k_end; k += 4)
        {
            __m128 oval = _mm_setzero_ps();
            for (int j = 0; j < n; j++)
            {
                float p_invscale = p_row[j] / vscales[j * num_blocks + vb];
                __m128 pvec = _mm_set1_ps(p_invscale);
                __m128i v8 = _mm_cvtsi32_si128(*(const int*)(V + j * out_d + k));
                __m128i v32 = _mm_cvtepi8_epi32(v8);
                __m128 vfp = _mm_cvtepi32_ps(v32);
                oval = _mm_add_ps(oval, _mm_mul_ps(pvec, vfp));
            }
            _mm_storeu_ps(out + k, oval);
        }
        for (; k < k_end; k++)
        {
            float sum = 0.f;
            for (int j = 0; j < n; j++)
                sum += p_row[j] / vscales[j * num_blocks + vb] * V[j * out_d + k];
            out[k] = sum;
        }
    }
}
#endif // __AVX2__

#if __AVX512F__
static inline void pv_float_int8_gemm_row_avx512_kernel(float* out, const float* p_row,
        const signed char* V, const float* vscales,
        int n, int out_d)
{
    const int num_blocks = (out_d + 31) / 32;
    for (int vb = 0; vb < num_blocks; vb++)
    {
        int k_start = vb * 32;
        int k_end = k_start + 32 < out_d ? k_start + 32 : out_d;
        int k = k_start;
        for (; k + 15 < k_end; k += 16)
        {
            __m512 oval = _mm512_setzero_ps();
            for (int j = 0; j < n; j++)
            {
                float p_invscale = p_row[j] / vscales[j * num_blocks + vb];
                __m512 pvec = _mm512_set1_ps(p_invscale);
                __m128i v8 = _mm_loadu_si128((const __m128i*)(V + j * out_d + k));
                __m512i v32 = _mm512_cvtepi8_epi32(v8);
                __m512 vfp = _mm512_cvtepi32_ps(v32);
                oval = _mm512_fmadd_ps(pvec, vfp, oval);
            }
            _mm512_storeu_ps(out + k, oval);
        }
        for (; k + 7 < k_end; k += 8)
        {
            __m256 oval = _mm256_setzero_ps();
            for (int j = 0; j < n; j++)
            {
                float p_invscale = p_row[j] / vscales[j * num_blocks + vb];
                __m256 pvec = _mm256_set1_ps(p_invscale);
                __m128i v8 = _mm_loadl_epi64((const __m128i*)(V + j * out_d + k));
                __m256i v32 = _mm256_cvtepi8_epi32(v8);
                __m256 vfp = _mm256_cvtepi32_ps(v32);
                oval = _mm256_fmadd_ps(pvec, vfp, oval);
            }
            _mm256_storeu_ps(out + k, oval);
        }
        for (; k < k_end; k++)
        {
            float sum = 0.f;
            for (int j = 0; j < n; j++)
                sum += p_row[j] / vscales[j * num_blocks + vb] * V[j * out_d + k];
            out[k] = sum;
        }
    }
}
#endif // __AVX512F__

static void pv_float_int8_gemm_row(float* out, const float* p_row,
                                   const signed char* V, const float* vscales,
                                   int n, int out_d)
{
#if __AVX512F__
    pv_float_int8_gemm_row_avx512_kernel(out, p_row, V, vscales, n, out_d);
#else
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        pv_float_int8_gemm_row_avx2(out, p_row, V, vscales, n, out_d);
        return;
    }
#endif
#if __AVX2__
    pv_float_int8_gemm_row_avx2(out, p_row, V, vscales, n, out_d);
#elif __SSE2__
    pv_float_int8_gemm_row_sse2_kernel(out, p_row, V, vscales, n, out_d);
#else
    pv_float_int8_gemm_row_scalar_kernel(out, p_row, V, vscales, n, out_d);
#endif
#endif
}

// ------------------- PV Float×Int8 FMA Block (online softmax) -------------------

static inline void pv_float_int8_fma_block_scalar_kernel(float* out, float p_invscale, const signed char* v, int len)
{
    for (int k = 0; k < len; k++)
        out[k] += p_invscale * v[k];
}

#if __SSE2__
static inline void pv_float_int8_fma_block_sse2_kernel(float* out, float p_invscale, const signed char* v, int len)
{
    __m128 pvec = _mm_set1_ps(p_invscale);
    int k = 0;
    for (; k + 3 < len; k += 4)
    {
        __m128i v8 = _mm_cvtsi32_si128(*(const int*)(v + k));
        __m128i v32 = _mm_cvtepi8_epi32(v8);
        __m128 vfp = _mm_cvtepi32_ps(v32);
        _mm_storeu_ps(out + k, _mm_add_ps(_mm_loadu_ps(out + k), _mm_mul_ps(pvec, vfp)));
    }
    for (; k < len; k++)
        out[k] += p_invscale * v[k];
}
#endif // __SSE2__

#if __AVX2__
inline void pv_float_int8_fma_block_avx2_kernel(float* out, float p_invscale, const signed char* v, int len)
{
    __m256 pvec = _mm256_set1_ps(p_invscale);
    int k = 0;
    for (; k + 7 < len; k += 8)
    {
        __m128i v8 = _mm_loadl_epi64((const __m128i*)(v + k));
        __m256i v32 = _mm256_cvtepi8_epi32(v8);
        __m256 vfp = _mm256_cvtepi32_ps(v32);
        _mm256_storeu_ps(out + k, _mm256_fmadd_ps(pvec, vfp, _mm256_loadu_ps(out + k)));
    }
    for (; k + 3 < len; k += 4)
    {
        __m128 pvec128 = _mm_set1_ps(p_invscale);
        __m128i v8 = _mm_cvtsi32_si128(*(const int*)(v + k));
        __m128i v32 = _mm_cvtepi8_epi32(v8);
        __m128 vfp = _mm_cvtepi32_ps(v32);
        _mm_storeu_ps(out + k, _mm_add_ps(_mm_loadu_ps(out + k), _mm_mul_ps(pvec128, vfp)));
    }
    for (; k < len; k++)
        out[k] += p_invscale * v[k];
}
#endif // __AVX2__

#if __AVX512F__
static inline void pv_float_int8_fma_block_avx512_kernel(float* out, float p_invscale, const signed char* v, int len)
{
    __m512 pvec = _mm512_set1_ps(p_invscale);
    int k = 0;
    for (; k + 15 < len; k += 16)
    {
        __m128i v8 = _mm_loadu_si128((const __m128i*)(v + k));
        __m512i v32 = _mm512_cvtepi8_epi32(v8);
        __m512 vfp = _mm512_cvtepi32_ps(v32);
        _mm512_storeu_ps(out + k, _mm512_fmadd_ps(pvec, vfp, _mm512_loadu_ps(out + k)));
    }
    for (; k + 7 < len; k += 8)
    {
        __m256 pvec256 = _mm256_set1_ps(p_invscale);
        __m128i v8 = _mm_loadl_epi64((const __m128i*)(v + k));
        __m256i v32 = _mm256_cvtepi8_epi32(v8);
        __m256 vfp = _mm256_cvtepi32_ps(v32);
        _mm256_storeu_ps(out + k, _mm256_fmadd_ps(pvec256, vfp, _mm256_loadu_ps(out + k)));
    }
    for (; k < len; k++)
        out[k] += p_invscale * v[k];
}
#endif // __AVX512F__

static void pv_float_int8_fma_block(float* out, float p_invscale, const signed char* v, int len)
{
#if __AVX512F__
    pv_float_int8_fma_block_avx512_kernel(out, p_invscale, v, len);
#else
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        pv_float_int8_fma_block_avx2(out, p_invscale, v, len);
        return;
    }
#endif
#if __AVX2__
    pv_float_int8_fma_block_avx2(out, p_invscale, v, len);
#elif __SSE2__
    pv_float_int8_fma_block_sse2_kernel(out, p_invscale, v, len);
#else
    pv_float_int8_fma_block_scalar_kernel(out, p_invscale, v, len);
#endif
#endif
}

#if __AVX2__
inline void pv_float_int8_gemm_tile_avx2_kernel(float* O, const float* P,
        const signed char* V, const float* vscales,
        int block_m, int block_n, int out_embed_dim)
{
    const int v_num_blocks = (out_embed_dim + 31) / 32;
    int i = 0;
    for (; i + 1 < block_m; i += 2)
    {
        const float* p0 = P + i * block_n;
        const float* p1 = P + (i + 1) * block_n;
        for (int vb = 0; vb < v_num_blocks; vb++)
        {
            int k_start = vb * 32;
            int k_end = k_start + 32 < out_embed_dim ? k_start + 32 : out_embed_dim;
            int len = k_end - k_start;
            if (len <= 0) continue;
            if (len == 32)
            {
                __m256 acc0_0 = _mm256_setzero_ps();
                __m256 acc0_1 = _mm256_setzero_ps();
                __m256 acc0_2 = _mm256_setzero_ps();
                __m256 acc0_3 = _mm256_setzero_ps();
                __m256 acc1_0 = _mm256_setzero_ps();
                __m256 acc1_1 = _mm256_setzero_ps();
                __m256 acc1_2 = _mm256_setzero_ps();
                __m256 acc1_3 = _mm256_setzero_ps();
                for (int j = 0; j < block_n; j++)
                {
                    float vscale = vscales[j * v_num_blocks + vb];
                    __m256 vscale8 = _mm256_set1_ps(vscale);
                    const signed char* vptr = V + j * out_embed_dim + k_start;
                    __m128i v8_0 = _mm_loadl_epi64((const __m128i*)(vptr + 0));
                    __m128i v8_1 = _mm_loadl_epi64((const __m128i*)(vptr + 8));
                    __m128i v8_2 = _mm_loadl_epi64((const __m128i*)(vptr + 16));
                    __m128i v8_3 = _mm_loadl_epi64((const __m128i*)(vptr + 24));
                    __m256 vval_0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v8_0)), vscale8);
                    __m256 vval_1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v8_1)), vscale8);
                    __m256 vval_2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v8_2)), vscale8);
                    __m256 vval_3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v8_3)), vscale8);
                    __m256 pvec0 = _mm256_set1_ps(p0[j]);
                    __m256 pvec1 = _mm256_set1_ps(p1[j]);
                    acc0_0 = _mm256_fmadd_ps(pvec0, vval_0, acc0_0);
                    acc0_1 = _mm256_fmadd_ps(pvec0, vval_1, acc0_1);
                    acc0_2 = _mm256_fmadd_ps(pvec0, vval_2, acc0_2);
                    acc0_3 = _mm256_fmadd_ps(pvec0, vval_3, acc0_3);
                    acc1_0 = _mm256_fmadd_ps(pvec1, vval_0, acc1_0);
                    acc1_1 = _mm256_fmadd_ps(pvec1, vval_1, acc1_1);
                    acc1_2 = _mm256_fmadd_ps(pvec1, vval_2, acc1_2);
                    acc1_3 = _mm256_fmadd_ps(pvec1, vval_3, acc1_3);
                }
                float* optr0 = O + i * out_embed_dim + k_start;
                float* optr1 = O + (i + 1) * out_embed_dim + k_start;
                _mm256_storeu_ps(optr0 + 0, _mm256_add_ps(_mm256_loadu_ps(optr0 + 0), acc0_0));
                _mm256_storeu_ps(optr0 + 8, _mm256_add_ps(_mm256_loadu_ps(optr0 + 8), acc0_1));
                _mm256_storeu_ps(optr0 + 16, _mm256_add_ps(_mm256_loadu_ps(optr0 + 16), acc0_2));
                _mm256_storeu_ps(optr0 + 24, _mm256_add_ps(_mm256_loadu_ps(optr0 + 24), acc0_3));
                _mm256_storeu_ps(optr1 + 0, _mm256_add_ps(_mm256_loadu_ps(optr1 + 0), acc1_0));
                _mm256_storeu_ps(optr1 + 8, _mm256_add_ps(_mm256_loadu_ps(optr1 + 8), acc1_1));
                _mm256_storeu_ps(optr1 + 16, _mm256_add_ps(_mm256_loadu_ps(optr1 + 16), acc1_2));
                _mm256_storeu_ps(optr1 + 24, _mm256_add_ps(_mm256_loadu_ps(optr1 + 24), acc1_3));
            }
            else
            {
                for (int j = 0; j < block_n; j++)
                {
                    float vscale = vscales[j * v_num_blocks + vb];
                    const signed char* vptr = V + j * out_embed_dim + k_start;
                    for (int k = 0; k < len; k++)
                    {
                        float vval = (float)vptr[k] * vscale;
                        O[i * out_embed_dim + k_start + k] += p0[j] * vval;
                        O[(i + 1) * out_embed_dim + k_start + k] += p1[j] * vval;
                    }
                }
            }
        }
    }
    for (; i < block_m; i++)
    {
        const float* p_row = P + i * block_n;
        for (int j = 0; j < block_n; j++)
        {
            float p = p_row[j];
            const signed char* vptr = V + j * out_embed_dim;
            const float* vscales_row = vscales + j * v_num_blocks;
            for (int vb = 0; vb < v_num_blocks; vb++)
            {
                int k_start = vb * 32;
                int k_end = k_start + 32 < out_embed_dim ? k_start + 32 : out_embed_dim;
                int len = k_end - k_start;
                if (len <= 0) continue;
                float vscale = vscales_row[vb];
                for (int k = 0; k < len; k++)
                {
                    O[i * out_embed_dim + k_start + k] += p * (float)vptr[k_start + k] * vscale;
                }
            }
        }
    }
}
#endif

static void pv_float_int8_gemm_tile(float* O, const float* P,
                                    const signed char* V, const float* vscales,
                                    int block_m, int block_n, int out_embed_dim)
{
    const int v_num_blocks = (out_embed_dim + 31) / 32;
#if __AVX512F__
    int i = 0;
    for (; i + 3 < block_m; i += 4)
    {
        const float* p0 = P + i * block_n;
        const float* p1 = P + (i + 1) * block_n;
        const float* p2 = P + (i + 2) * block_n;
        const float* p3 = P + (i + 3) * block_n;
        for (int vb = 0; vb < v_num_blocks; vb++)
        {
            int k_start = vb * 32;
            int k_end = k_start + 32 < out_embed_dim ? k_start + 32 : out_embed_dim;
            int len = k_end - k_start;
            if (len <= 0) continue;
            if (len == 32)
            {
                __m512 acc0_0 = _mm512_setzero_ps();
                __m512 acc0_1 = _mm512_setzero_ps();
                __m512 acc1_0 = _mm512_setzero_ps();
                __m512 acc1_1 = _mm512_setzero_ps();
                __m512 acc2_0 = _mm512_setzero_ps();
                __m512 acc2_1 = _mm512_setzero_ps();
                __m512 acc3_0 = _mm512_setzero_ps();
                __m512 acc3_1 = _mm512_setzero_ps();
                for (int j = 0; j < block_n; j++)
                {
                    float vscale = vscales[j * v_num_blocks + vb];
                    __m512 vscale16 = _mm512_set1_ps(vscale);
                    const signed char* vptr = V + j * out_embed_dim + k_start;
                    __m128i v8_0 = _mm_loadu_si128((const __m128i*)(vptr + 0));
                    __m128i v8_1 = _mm_loadu_si128((const __m128i*)(vptr + 16));
                    __m512 vval_0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v8_0)), vscale16);
                    __m512 vval_1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v8_1)), vscale16);
                    __m512 pvec0 = _mm512_set1_ps(p0[j]);
                    __m512 pvec1 = _mm512_set1_ps(p1[j]);
                    __m512 pvec2 = _mm512_set1_ps(p2[j]);
                    __m512 pvec3 = _mm512_set1_ps(p3[j]);
                    acc0_0 = _mm512_fmadd_ps(pvec0, vval_0, acc0_0);
                    acc0_1 = _mm512_fmadd_ps(pvec0, vval_1, acc0_1);
                    acc1_0 = _mm512_fmadd_ps(pvec1, vval_0, acc1_0);
                    acc1_1 = _mm512_fmadd_ps(pvec1, vval_1, acc1_1);
                    acc2_0 = _mm512_fmadd_ps(pvec2, vval_0, acc2_0);
                    acc2_1 = _mm512_fmadd_ps(pvec2, vval_1, acc2_1);
                    acc3_0 = _mm512_fmadd_ps(pvec3, vval_0, acc3_0);
                    acc3_1 = _mm512_fmadd_ps(pvec3, vval_1, acc3_1);
                }
                float* optr0 = O + i * out_embed_dim + k_start;
                float* optr1 = O + (i + 1) * out_embed_dim + k_start;
                float* optr2 = O + (i + 2) * out_embed_dim + k_start;
                float* optr3 = O + (i + 3) * out_embed_dim + k_start;
                _mm512_storeu_ps(optr0 + 0, _mm512_add_ps(_mm512_loadu_ps(optr0 + 0), acc0_0));
                _mm512_storeu_ps(optr0 + 16, _mm512_add_ps(_mm512_loadu_ps(optr0 + 16), acc0_1));
                _mm512_storeu_ps(optr1 + 0, _mm512_add_ps(_mm512_loadu_ps(optr1 + 0), acc1_0));
                _mm512_storeu_ps(optr1 + 16, _mm512_add_ps(_mm512_loadu_ps(optr1 + 16), acc1_1));
                _mm512_storeu_ps(optr2 + 0, _mm512_add_ps(_mm512_loadu_ps(optr2 + 0), acc2_0));
                _mm512_storeu_ps(optr2 + 16, _mm512_add_ps(_mm512_loadu_ps(optr2 + 16), acc2_1));
                _mm512_storeu_ps(optr3 + 0, _mm512_add_ps(_mm512_loadu_ps(optr3 + 0), acc3_0));
                _mm512_storeu_ps(optr3 + 16, _mm512_add_ps(_mm512_loadu_ps(optr3 + 16), acc3_1));
            }
            else
            {
                for (int j = 0; j < block_n; j++)
                {
                    float vscale = vscales[j * v_num_blocks + vb];
                    const signed char* vptr = V + j * out_embed_dim + k_start;
                    for (int k = 0; k < len; k++)
                    {
                        float vval = (float)vptr[k] * vscale;
                        O[i * out_embed_dim + k_start + k] += p0[j] * vval;
                        O[(i + 1) * out_embed_dim + k_start + k] += p1[j] * vval;
                        O[(i + 2) * out_embed_dim + k_start + k] += p2[j] * vval;
                        O[(i + 3) * out_embed_dim + k_start + k] += p3[j] * vval;
                    }
                }
            }
        }
    }
    for (; i < block_m; i++)
    {
        const float* p_row = P + i * block_n;
        for (int vb = 0; vb < v_num_blocks; vb++)
        {
            int k_start = vb * 32;
            int k_end = k_start + 32 < out_embed_dim ? k_start + 32 : out_embed_dim;
            int len = k_end - k_start;
            if (len <= 0) continue;
            if (len == 32)
            {
                __m512 acc0 = _mm512_setzero_ps();
                __m512 acc1 = _mm512_setzero_ps();
                for (int j = 0; j < block_n; j++)
                {
                    float vscale = vscales[j * v_num_blocks + vb];
                    __m512 vscale16 = _mm512_set1_ps(vscale);
                    const signed char* vptr = V + j * out_embed_dim + k_start;
                    __m128i v8_0 = _mm_loadu_si128((const __m128i*)(vptr + 0));
                    __m128i v8_1 = _mm_loadu_si128((const __m128i*)(vptr + 16));
                    __m512 vval_0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v8_0)), vscale16);
                    __m512 vval_1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v8_1)), vscale16);
                    __m512 pvec = _mm512_set1_ps(p_row[j]);
                    acc0 = _mm512_fmadd_ps(pvec, vval_0, acc0);
                    acc1 = _mm512_fmadd_ps(pvec, vval_1, acc1);
                }
                float* optr = O + i * out_embed_dim + k_start;
                _mm512_storeu_ps(optr + 0, _mm512_add_ps(_mm512_loadu_ps(optr + 0), acc0));
                _mm512_storeu_ps(optr + 16, _mm512_add_ps(_mm512_loadu_ps(optr + 16), acc1));
            }
            else
            {
                for (int j = 0; j < block_n; j++)
                {
                    float p = p_row[j];
                    float vscale = vscales[j * v_num_blocks + vb];
                    const signed char* vptr = V + j * out_embed_dim + k_start;
                    for (int k = 0; k < len; k++)
                    {
                        O[i * out_embed_dim + k_start + k] += p * (float)vptr[k] * vscale;
                    }
                }
            }
        }
    }
#else
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        pv_float_int8_gemm_tile_avx2(O, P, V, vscales, block_m, block_n, out_embed_dim);
        return;
    }
#endif
#if __AVX2__
    int i = 0;
    for (; i + 1 < block_m; i += 2)
    {
        const float* p0 = P + i * block_n;
        const float* p1 = P + (i + 1) * block_n;
        for (int vb = 0; vb < v_num_blocks; vb++)
        {
            int k_start = vb * 32;
            int k_end = k_start + 32 < out_embed_dim ? k_start + 32 : out_embed_dim;
            int len = k_end - k_start;
            if (len <= 0) continue;
            if (len == 32)
            {
                __m256 acc0_0 = _mm256_setzero_ps();
                __m256 acc0_1 = _mm256_setzero_ps();
                __m256 acc0_2 = _mm256_setzero_ps();
                __m256 acc0_3 = _mm256_setzero_ps();
                __m256 acc1_0 = _mm256_setzero_ps();
                __m256 acc1_1 = _mm256_setzero_ps();
                __m256 acc1_2 = _mm256_setzero_ps();
                __m256 acc1_3 = _mm256_setzero_ps();
                for (int j = 0; j < block_n; j++)
                {
                    float vscale = vscales[j * v_num_blocks + vb];
                    __m256 vscale8 = _mm256_set1_ps(vscale);
                    const signed char* vptr = V + j * out_embed_dim + k_start;
                    __m128i v8_0 = _mm_loadl_epi64((const __m128i*)(vptr + 0));
                    __m128i v8_1 = _mm_loadl_epi64((const __m128i*)(vptr + 8));
                    __m128i v8_2 = _mm_loadl_epi64((const __m128i*)(vptr + 16));
                    __m128i v8_3 = _mm_loadl_epi64((const __m128i*)(vptr + 24));
                    __m256 vval_0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v8_0)), vscale8);
                    __m256 vval_1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v8_1)), vscale8);
                    __m256 vval_2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v8_2)), vscale8);
                    __m256 vval_3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v8_3)), vscale8);
                    __m256 pvec0 = _mm256_set1_ps(p0[j]);
                    __m256 pvec1 = _mm256_set1_ps(p1[j]);
                    acc0_0 = _mm256_fmadd_ps(pvec0, vval_0, acc0_0);
                    acc0_1 = _mm256_fmadd_ps(pvec0, vval_1, acc0_1);
                    acc0_2 = _mm256_fmadd_ps(pvec0, vval_2, acc0_2);
                    acc0_3 = _mm256_fmadd_ps(pvec0, vval_3, acc0_3);
                    acc1_0 = _mm256_fmadd_ps(pvec1, vval_0, acc1_0);
                    acc1_1 = _mm256_fmadd_ps(pvec1, vval_1, acc1_1);
                    acc1_2 = _mm256_fmadd_ps(pvec1, vval_2, acc1_2);
                    acc1_3 = _mm256_fmadd_ps(pvec1, vval_3, acc1_3);
                }
                float* optr0 = O + i * out_embed_dim + k_start;
                float* optr1 = O + (i + 1) * out_embed_dim + k_start;
                _mm256_storeu_ps(optr0 + 0, _mm256_add_ps(_mm256_loadu_ps(optr0 + 0), acc0_0));
                _mm256_storeu_ps(optr0 + 8, _mm256_add_ps(_mm256_loadu_ps(optr0 + 8), acc0_1));
                _mm256_storeu_ps(optr0 + 16, _mm256_add_ps(_mm256_loadu_ps(optr0 + 16), acc0_2));
                _mm256_storeu_ps(optr0 + 24, _mm256_add_ps(_mm256_loadu_ps(optr0 + 24), acc0_3));
                _mm256_storeu_ps(optr1 + 0, _mm256_add_ps(_mm256_loadu_ps(optr1 + 0), acc1_0));
                _mm256_storeu_ps(optr1 + 8, _mm256_add_ps(_mm256_loadu_ps(optr1 + 8), acc1_1));
                _mm256_storeu_ps(optr1 + 16, _mm256_add_ps(_mm256_loadu_ps(optr1 + 16), acc1_2));
                _mm256_storeu_ps(optr1 + 24, _mm256_add_ps(_mm256_loadu_ps(optr1 + 24), acc1_3));
            }
            else
            {
                for (int j = 0; j < block_n; j++)
                {
                    float vscale = vscales[j * v_num_blocks + vb];
                    const signed char* vptr = V + j * out_embed_dim + k_start;
                    for (int k = 0; k < len; k++)
                    {
                        float vval = (float)vptr[k] * vscale;
                        O[i * out_embed_dim + k_start + k] += p0[j] * vval;
                        O[(i + 1) * out_embed_dim + k_start + k] += p1[j] * vval;
                    }
                }
            }
        }
    }
    for (; i < block_m; i++)
    {
        const float* p_row = P + i * block_n;
        for (int j = 0; j < block_n; j++)
        {
            float p = p_row[j];
            const signed char* vptr = V + j * out_embed_dim;
            const float* vscales_row = vscales + j * v_num_blocks;
            for (int vb = 0; vb < v_num_blocks; vb++)
            {
                int k_start = vb * 32;
                int k_end = k_start + 32 < out_embed_dim ? k_start + 32 : out_embed_dim;
                int len = k_end - k_start;
                if (len <= 0) continue;
                float vscale = vscales_row[vb];
                for (int k = 0; k < len; k++)
                {
                    O[i * out_embed_dim + k_start + k] += p * (float)vptr[k_start + k] * vscale;
                }
            }
        }
    }
#elif __SSE2__
    for (int j = 0; j < block_n; j++)
    {
        const signed char* vptr = V + j * out_embed_dim;
        const float* vscales_row = vscales + j * v_num_blocks;
        for (int vb = 0; vb < v_num_blocks; vb++)
        {
            int k_start = vb * 32;
            int k_end = k_start + 32 < out_embed_dim ? k_start + 32 : out_embed_dim;
            int len = k_end - k_start;
            if (len <= 0) continue;
            float vscale = vscales_row[vb];
            int k = k_start;
            for (; k + 3 < k_end; k += 4)
            {
                __m128i v8 = _mm_cvtsi32_si128(*(const int*)(vptr + k));
                __m128i v32 = _mm_cvtepi8_epi32(v8);
                __m128 vfp = _mm_cvtepi32_ps(v32);
                __m128 vval = _mm_mul_ps(vfp, _mm_set1_ps(vscale));
                for (int ii = 0; ii < block_m; ii++)
                {
                    float p = P[ii * block_n + j];
                    __m128 pvec = _mm_set1_ps(p);
                    float* optr = O + ii * out_embed_dim + k;
                    _mm_storeu_ps(optr, _mm_add_ps(_mm_loadu_ps(optr), _mm_mul_ps(pvec, vval)));
                }
            }
            for (; k < k_end; k++)
            {
                float vval = (float)vptr[k] * vscale;
                for (int ii = 0; ii < block_m; ii++)
                {
                    O[ii * out_embed_dim + k] += P[ii * block_n + j] * vval;
                }
            }
        }
    }
#else
    for (int j = 0; j < block_n; j++)
    {
        const signed char* vptr = V + j * out_embed_dim;
        const float* vscales_row = vscales + j * v_num_blocks;
        for (int vb = 0; vb < v_num_blocks; vb++)
        {
            int k_start = vb * 32;
            int k_end = k_start + 32 < out_embed_dim ? k_start + 32 : out_embed_dim;
            int len = k_end - k_start;
            if (len <= 0) continue;
            float vscale = vscales_row[vb];
            for (int k = k_start; k < k_end; k++)
            {
                float vval = (float)vptr[k] * vscale;
                for (int ii = 0; ii < block_m; ii++)
                {
                    O[ii * out_embed_dim + k] += P[ii * block_n + j] * vval;
                }
            }
        }
    }
#endif
#endif
}

#if __SSE2__ && !__SSSE3__
#undef _mm_cvtepi8_epi16
#endif
#if __SSE2__ && !__SSE4_1__
#undef _mm_cvtepi8_epi32
#endif

#endif // SDPA_X86_INT8_H
