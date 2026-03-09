// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// This file is #include'd inside a named namespace in:
//   - binaryop_x86.cpp               (namespace BinaryOp_x86_functor)
//   - binaryop_x86_avx512bf16.cpp    (namespace BinaryOp_x86_functor)
// Do NOT add a namespace here.
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

struct binary_op_pow
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return (float)powf(x, y);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return pow_ps(x, y);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return pow256_ps(x, y);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return pow512_ps(x, y);
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
        return pow_ps(y, x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return pow256_ps(y, x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return pow512_ps(y, x);
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
