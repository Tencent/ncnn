// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

struct reduction_op_add
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return x + y;
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_add_ps(x, y);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_add_ps(x, y);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_add_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct reduction_op_mul
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return x * y;
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_mul_ps(x, y);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_mul_ps(x, y);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_mul_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct reduction_op_max
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return x > y ? x : y;
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_max_ps(x, y);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_max_ps(x, y);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_max_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct reduction_op_min
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return x < y ? x : y;
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_min_ps(x, y);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_min_ps(x, y);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_min_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct reduction_op_asum
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return x + fabsf(y);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_add_ps(x, abs_ps(y));
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_add_ps(x, abs256_ps(y));
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_add_ps(x, abs512_ps(y));
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct reduction_op_sumsq
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return x + y * y;
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_comp_fmadd_ps(y, y, x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_comp_fmadd_ps(y, y, x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_fmadd_ps(y, y, x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct reduction_op_sumexp
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return x + expf(y);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_add_ps(x, exp_ps(y));
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_add_ps(x, exp256_ps(y));
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_add_ps(x, exp512_ps(y));
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};
