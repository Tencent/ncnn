// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

struct binary_op_add
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

struct binary_op_sub
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return x - y;
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_sub_ps(x, y);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_sub_ps(x, y);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_sub_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_mul
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

struct binary_op_div
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return x / y;
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_div_ps(x, y);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_div_ps(x, y);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_div_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_max
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return std::max(x, y);
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

struct binary_op_min
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return std::min(x, y);
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

#if __SSE2__
// Keep the libm calls and their stack storage out of the vector loops.
template<typename T>
#if defined(_MSC_VER)
__declspec(noinline)
#elif defined(__GNUC__)
__attribute__((noinline))
#endif
static T binary_op_pow_fallback(const T& x, const T& y, const T& safe_base, int positive_mask, T (*pow_positive)(const T&, const T&))
{
    const int lanes = sizeof(T) / sizeof(float);
    float bases[lanes];
    float exponents[lanes];
    float values[lanes];
    memcpy(bases, &x, sizeof(T));
    memcpy(exponents, &y, sizeof(T));
    T result;
    if (positive_mask)
    {
        result = pow_positive(safe_base, y);
        memcpy(values, &result, sizeof(T));
    }
    for (int i = 0; i < lanes; i++)
    {
        if (!(positive_mask & (1 << i)))
            values[i] = powf(bases[i], exponents[i]);
    }
    memcpy(&result, values, sizeof(T));
    return result;
}
#endif // __SSE2__

struct binary_op_pow
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return (float)powf(x, y);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        const __m128 positive = _mm_cmpgt_ps(x, _mm_setzero_ps());
        const int mask = _mm_movemask_ps(positive);
        if (mask == 0xf)
            return pow_ps(x, y);

        // Give log/exp a valid base with a nonzero logarithm in the fallback lanes.
        const __m128 base = _mm_or_ps(_mm_and_ps(positive, x), _mm_andnot_ps(positive, _mm_set1_ps(2.f)));
        return binary_op_pow_fallback(x, y, base, mask, pow_ps);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        const __m256 positive = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_GT_OQ);
        const int mask = _mm256_movemask_ps(positive);
        if (mask == 0xff)
            return pow256_ps(x, y);

        const __m256 base = _mm256_blendv_ps(_mm256_set1_ps(2.f), x, positive);
        return binary_op_pow_fallback(x, y, base, mask, pow256_ps);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        const __mmask16 mask = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_GT_OQ);
        if (mask == 0xffff)
            return pow512_ps(x, y);

        const __m512 base = _mm512_mask_mov_ps(_mm512_set1_ps(2.f), mask, x);
        return binary_op_pow_fallback(x, y, base, mask, pow512_ps);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_rsub
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return y - x;
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_sub_ps(y, x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_sub_ps(y, x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_sub_ps(y, x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_rdiv
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return y / x;
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_div_ps(y, x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_div_ps(y, x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_div_ps(y, x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_rpow
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return (float)powf(y, x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return binary_op_pow().func_pack4(y, x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return binary_op_pow().func_pack8(y, x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return binary_op_pow().func_pack16(y, x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_atan2
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return (float)atan2f(x, y);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return atan2_ps(x, y);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return atan2256_ps(x, y);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return atan2512_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_ratan2
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return (float)atan2f(y, x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return atan2_ps(y, x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return atan2256_ps(y, x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return atan2512_ps(y, x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_fmod
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return (float)fmodf(x, y);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return fmod_ps(x, y);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return fmod256_ps(x, y);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return fmod512_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_rfmod
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return (float)fmodf(y, x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return fmod_ps(y, x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return fmod256_ps(y, x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return fmod512_ps(y, x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_logaddexp
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        float max_xy = std::max(x, y);
        float min_xy = std::min(x, y);
        return (float)(max_xy + log1pf(expf(min_xy - max_xy)));
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return logaddexp_ps(x, y);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return logaddexp256_ps(x, y);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return logaddexp512_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_floor_divide
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return (float)floorf(x / y);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return floor_divide_ps(x, y);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return floor_divide256_ps(x, y);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return floor_divide512_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_rfloor_divide
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return (float)floorf(y / x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return floor_divide_ps(y, x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return floor_divide256_ps(y, x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return floor_divide512_ps(y, x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_remainder
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return (float)remainderf(x, y);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return remainder_ps(x, y);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return remainder256_ps(x, y);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return remainder512_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_rremainder
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return (float)remainderf(y, x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return remainder_ps(y, x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return remainder256_ps(y, x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return remainder512_ps(y, x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};
