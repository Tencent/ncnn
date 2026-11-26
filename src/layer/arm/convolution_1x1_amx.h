// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// 1x1 stride1 convolution implemented with Apple AMX
// C[num_output][pixels] = weight[num_output][num_input] * bottom[num_input][pixels]

#if defined __APPLE__ && __aarch64__

#include "apple_amx.h"

// generic fp32 output tile loop
// A: K x M transposed weight, B: K x N input, Btail: K x 16 zero padded tail columns
// out: M x N packed output
static void conv1x1s1_apple_amx_gemm_f32(const float* A, int lda, const float* B, int ldb, const float* Btail, float* outptr, size_t out_cstep, int M, int N, int K, const float* bias, int out_elempack, const Option& opt)
{
    const int M16 = M - (M % 16);
    const int N16 = N - (N % 16);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int m0 = 0; m0 < M16; m0 += 16)
    {
        NCNN_AMX_SET();

        for (int n0 = 0; n0 < N16; n0 += 16)
        {
            ncnn_apple_amx_gemm_tile16(A, lda, B, ldb, m0, n0, K);

            if (out_elempack == 1)
            {
                for (int j = 0; j < 16; j++)
                {
                    const int m = m0 + j;
                    float* dst = outptr + (size_t)m * out_cstep + n0;

                    ncnn_apple_amx_store_row16(j, dst);

                    if (bias)
                    {
                        const float b = bias[m];
                        for (int i = 0; i < 16; i++)
                            dst[i] += b;
                    }
                }
            }
            else
            {
                // gather z rows then scatter to the packed channel layout
                float tmp[16][16];
                for (int j = 0; j < 16; j++)
                {
                    ncnn_apple_amx_store_row16(j, &tmp[j][0]);
                }

                for (int jg = 0; jg < 16; jg += out_elempack)
                {
                    for (int n = 0; n < 16; n++)
                    {
                        float* dst = outptr + (size_t)((m0 + jg) / out_elempack) * out_cstep + (size_t)(n0 + n) * out_elempack;

                        for (int i = 0; i < out_elempack; i++)
                        {
                            dst[i] = tmp[jg + i][n] + (bias ? bias[m0 + jg + i] : 0.f);
                        }
                    }
                }
            }
        }

        // remaining pixels as one zero padded 16-wide tile
        if (N16 < N)
        {
            ncnn_apple_amx_gemm_tile16(A, lda, Btail, 16, m0, 0, K);

            float tmp[16][16];
            for (int j = 0; j < 16; j++)
            {
                ncnn_apple_amx_store_row16(j, &tmp[j][0]);
            }

            const int nt = N - N16;

            if (out_elempack == 1)
            {
                for (int j = 0; j < 16; j++)
                {
                    const int m = m0 + j;
                    float* dst = outptr + (size_t)m * out_cstep + N16;
                    const float b = bias ? bias[m] : 0.f;

                    for (int n = 0; n < nt; n++)
                        dst[n] = tmp[j][n] + b;
                }
            }
            else
            {
                for (int jg = 0; jg < 16; jg += out_elempack)
                {
                    for (int n = 0; n < nt; n++)
                    {
                        float* dst = outptr + (size_t)((m0 + jg) / out_elempack) * out_cstep + (size_t)(N16 + n) * out_elempack;

                        for (int i = 0; i < out_elempack; i++)
                        {
                            dst[i] = tmp[jg + i][n] + (bias ? bias[m0 + jg + i] : 0.f);
                        }
                    }
                }
            }
        }

        NCNN_AMX_CLR();
    }

    // remaining output channels
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int m = M16; m < M; m++)
    {
        const float b = bias ? bias[m] : 0.f;

        for (int n = 0; n < N; n++)
        {
            float s = b;

            for (int k = 0; k < K; k++)
            {
                s += A[(size_t)k * lda + m] * B[(size_t)k * ldb + n];
            }

            if (out_elempack == 1)
                outptr[(size_t)m * out_cstep + n] = s;
            else
                outptr[(size_t)(m / out_elempack) * out_cstep + (size_t)n * out_elempack + (m % out_elempack)] = s;
        }
    }
}

// pack the fp32 input as a K x N matrix
static void conv1x1s1_apple_amx_pack_f32(const Mat& bottom_blob, Mat& Bpack, int K, int N, const Option& opt)
{
    const int elempack = bottom_blob.elempack;

    Bpack.create(K * N, (size_t)4u, opt.workspace_allocator);
    if (Bpack.empty())
        return;

    float* bp = Bpack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < bottom_blob.c; q++)
    {
        const float* p = bottom_blob.channel(q);

        for (int i = 0; i < elempack; i++)
        {
            float* dst = bp + (size_t)(q * elempack + i) * N;
            const float* src = p + i;

            for (int n = 0; n < N; n++)
            {
                dst[n] = src[n * elempack];
            }
        }
    }
}

static int conv1x1s1_apple_amx(const Mat& bottom_blob, Mat& top_blob, const Mat& _weight_amx, const Mat& _bias, const Option& opt)
{
    const int N = (int)top_blob.w * top_blob.h;

    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const int M = (int)top_blob.c * out_elempack;
    const int K = (int)bottom_blob.c * elempack;

    const float* A = _weight_amx; // K x M, transposed weight
    const int lda = M;
    const float* bias = _bias;

    // pack1 input is already N-major so it can be used in place
    Mat Bpack;
    const float* B = 0;
    int ldb = 0;

    if (elempack == 1)
    {
        B = bottom_blob;
        ldb = (int)(bottom_blob.cstep * elempack);
    }
    else
    {
        conv1x1s1_apple_amx_pack_f32(bottom_blob, Bpack, K, N, opt);
        if (Bpack.empty())
            return -100;

        B = Bpack;
        ldb = N;
    }

    // cstep counts packed elements, convert to float offset
    const size_t out_cstep = top_blob.cstep * out_elempack;

    // zero padded tail columns so the last pixel tile runs on AMX instead of scalar code
    Mat Btail;
    const float* Bt = 0;
    const int ntail = N % 16;
    if (ntail)
    {
        Btail.create(16 * K, (size_t)4u, opt.workspace_allocator);
        if (Btail.empty())
            return -100;

        Btail.fill(0.f);
        float* bt = Btail;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int k = 0; k < K; k++)
        {
            float* dst = bt + (size_t)k * 16;
            const float* src = B + (size_t)k * ldb + (N - ntail);

            for (int n = 0; n < ntail; n++)
            {
                dst[n] = src[n];
            }
        }

        Bt = Btail;
    }

    conv1x1s1_apple_amx_gemm_f32(A, lda, B, ldb, Bt, (float*)top_blob, out_cstep, M, N, K, bias, out_elempack, opt);

    return 0;
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

// fma16 widen computes a 32x32 output tile with 1024 mac per instruction
// A: K x lda fp16 transposed weight, lda is a multiple of 32 and the tail is zero padded
// B: K x ldb fp16 input, 32 contiguous fp16 along N
// z physical row r carries logical row r >> 1, even n when r is even
static inline void conv1x1s1_apple_amx_tile_f16(const __fp16* A, int lda, const __fp16* B, int ldb, int m0, int n0, int K, float* zt)
{
    for (int k = 0; k < K; k++)
    {
        NCNN_AMX_LDX((uint64_t)(B + (size_t)k * ldb + n0));
        NCNN_AMX_LDY((uint64_t)(A + (size_t)k * lda + m0));
        NCNN_AMX_FMA16((1ull << 62) | (k == 0 ? (1ull << 27) : 0ull));
    }

    // dump all 64 physical z rows, 16 f32 each
    for (int r = 0; r < 64; r++)
    {
        NCNN_AMX_STZ((uint64_t)(zt + r * 16) + ((uint64_t)r << 56));
    }
}

// interleave the even and odd z lanes of one logical row and convert to fp16 with bias
static inline void conv1x1s1_apple_amx_store_row_f16(const float* even, const float* odd, float b, __fp16* dst)
{
    float32x4_t bv = vdupq_n_f32(b);

    float16x8x2_t v0;
    v0.val[0] = vcombine_f16(vcvt_f16_f32(vaddq_f32(vld1q_f32(even), bv)), vcvt_f16_f32(vaddq_f32(vld1q_f32(even + 4), bv)));
    v0.val[1] = vcombine_f16(vcvt_f16_f32(vaddq_f32(vld1q_f32(odd), bv)), vcvt_f16_f32(vaddq_f32(vld1q_f32(odd + 4), bv)));
    vst2q_f16(dst, v0);

    float16x8x2_t v1;
    v1.val[0] = vcombine_f16(vcvt_f16_f32(vaddq_f32(vld1q_f32(even + 8), bv)), vcvt_f16_f32(vaddq_f32(vld1q_f32(even + 12), bv)));
    v1.val[1] = vcombine_f16(vcvt_f16_f32(vaddq_f32(vld1q_f32(odd + 8), bv)), vcvt_f16_f32(vaddq_f32(vld1q_f32(odd + 12), bv)));
    vst2q_f16(dst + 16, v1);
}

// transpose an 8x8 fp16 block, column i is stored at dst + i * dst_stride
static inline void conv1x1s1_apple_amx_transpose8x8_f16(const __fp16* src, int src_stride, __fp16* dst, int dst_stride)
{
    const unsigned short* s = (const unsigned short*)src;
    unsigned short* d = (unsigned short*)dst;

    uint16x8_t r0 = vld1q_u16(s);
    uint16x8_t r1 = vld1q_u16(s + src_stride);
    uint16x8_t r2 = vld1q_u16(s + src_stride * 2);
    uint16x8_t r3 = vld1q_u16(s + src_stride * 3);
    uint16x8_t r4 = vld1q_u16(s + src_stride * 4);
    uint16x8_t r5 = vld1q_u16(s + src_stride * 5);
    uint16x8_t r6 = vld1q_u16(s + src_stride * 6);
    uint16x8_t r7 = vld1q_u16(s + src_stride * 7);

    uint16x8_t t0 = vtrn1q_u16(r0, r1);
    uint16x8_t t1 = vtrn2q_u16(r0, r1);
    uint16x8_t t2 = vtrn1q_u16(r2, r3);
    uint16x8_t t3 = vtrn2q_u16(r2, r3);
    uint16x8_t t4 = vtrn1q_u16(r4, r5);
    uint16x8_t t5 = vtrn2q_u16(r4, r5);
    uint16x8_t t6 = vtrn1q_u16(r6, r7);
    uint16x8_t t7 = vtrn2q_u16(r6, r7);

    uint16x8_t u0 = vreinterpretq_u16_u32(vzip1q_u32(vreinterpretq_u32_u16(t0), vreinterpretq_u32_u16(t2)));
    uint16x8_t u1 = vreinterpretq_u16_u32(vzip2q_u32(vreinterpretq_u32_u16(t0), vreinterpretq_u32_u16(t2)));
    uint16x8_t u2 = vreinterpretq_u16_u32(vzip1q_u32(vreinterpretq_u32_u16(t1), vreinterpretq_u32_u16(t3)));
    uint16x8_t u3 = vreinterpretq_u16_u32(vzip2q_u32(vreinterpretq_u32_u16(t1), vreinterpretq_u32_u16(t3)));
    uint16x8_t u4 = vreinterpretq_u16_u32(vzip1q_u32(vreinterpretq_u32_u16(t4), vreinterpretq_u32_u16(t6)));
    uint16x8_t u5 = vreinterpretq_u16_u32(vzip2q_u32(vreinterpretq_u32_u16(t4), vreinterpretq_u32_u16(t6)));
    uint16x8_t u6 = vreinterpretq_u16_u32(vzip1q_u32(vreinterpretq_u32_u16(t5), vreinterpretq_u32_u16(t7)));
    uint16x8_t u7 = vreinterpretq_u16_u32(vzip2q_u32(vreinterpretq_u32_u16(t5), vreinterpretq_u32_u16(t7)));

    vst1q_u16(d + dst_stride * 0, vcombine_u16(vget_low_u16(u0), vget_low_u16(u4)));
    vst1q_u16(d + dst_stride * 1, vcombine_u16(vget_low_u16(u2), vget_low_u16(u6)));
    vst1q_u16(d + dst_stride * 2, vcombine_u16(vget_high_u16(u0), vget_high_u16(u4)));
    vst1q_u16(d + dst_stride * 3, vcombine_u16(vget_high_u16(u2), vget_high_u16(u6)));
    vst1q_u16(d + dst_stride * 4, vcombine_u16(vget_low_u16(u1), vget_low_u16(u5)));
    vst1q_u16(d + dst_stride * 5, vcombine_u16(vget_low_u16(u3), vget_low_u16(u7)));
    vst1q_u16(d + dst_stride * 6, vcombine_u16(vget_high_u16(u1), vget_high_u16(u5)));
    vst1q_u16(d + dst_stride * 7, vcombine_u16(vget_high_u16(u3), vget_high_u16(u7)));
}

// transpose a 4x4 fp16 block, column i is stored at dst + i * dst_stride
static inline void conv1x1s1_apple_amx_transpose4x4_f16(const __fp16* src, int src_stride, __fp16* dst, int dst_stride)
{
    const unsigned short* s = (const unsigned short*)src;
    unsigned short* d = (unsigned short*)dst;

    uint16x8_t ab = vcombine_u16(vld1_u16(s), vld1_u16(s + src_stride));
    uint16x8_t cd = vcombine_u16(vld1_u16(s + src_stride * 2), vld1_u16(s + src_stride * 3));

    uint16x8_t t0 = vtrn1q_u16(ab, cd);
    uint16x8_t t1 = vtrn2q_u16(ab, cd);

    uint16x8_t u0 = vreinterpretq_u16_u32(vzip1q_u32(vreinterpretq_u32_u16(t0), vreinterpretq_u32_u16(t1)));
    uint16x8_t u1 = vreinterpretq_u16_u32(vzip2q_u32(vreinterpretq_u32_u16(t0), vreinterpretq_u32_u16(t1)));

    vst1_u16(d + dst_stride * 0, vzip1_u16(vget_low_u16(u0), vget_low_u16(u1)));
    vst1_u16(d + dst_stride * 1, vzip2_u16(vget_low_u16(u0), vget_low_u16(u1)));
    vst1_u16(d + dst_stride * 2, vzip1_u16(vget_high_u16(u0), vget_high_u16(u1)));
    vst1_u16(d + dst_stride * 3, vzip2_u16(vget_high_u16(u0), vget_high_u16(u1)));
}

// repack the fp16 input as a K x N matrix with N contiguous
static void conv1x1s1_apple_amx_pack_f16(const Mat& bottom_blob, Mat& Bpack, int K, int N, const Option& opt)
{
    const int elempack = bottom_blob.elempack;

    Bpack.create(K * N, (size_t)2u, opt.workspace_allocator);
    if (Bpack.empty())
        return;

    __fp16* bp = (__fp16*)Bpack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < bottom_blob.c; q++)
    {
        const __fp16* p = (const __fp16*)bottom_blob.channel(q);
        __fp16* dst0 = bp + (size_t)(q * elempack) * N;

        if (elempack == 8)
        {
            int n = 0;
            for (; n + 8 <= N; n += 8)
            {
                conv1x1s1_apple_amx_transpose8x8_f16(p + (size_t)n * 8, 8, dst0 + n, N);
            }
            for (; n < N; n++)
            {
                for (int i = 0; i < 8; i++)
                    dst0[(size_t)i * N + n] = p[(size_t)n * 8 + i];
            }
        }
        else if (elempack == 4)
        {
            int n = 0;
            for (; n + 4 <= N; n += 4)
            {
                conv1x1s1_apple_amx_transpose4x4_f16(p + (size_t)n * 4, 4, dst0 + n, N);
            }
            for (; n < N; n++)
            {
                for (int i = 0; i < 4; i++)
                    dst0[(size_t)i * N + n] = p[(size_t)n * 4 + i];
            }
        }
        else
        {
            for (int n = 0; n < N; n++)
            {
                for (int i = 0; i < elempack; i++)
                    dst0[(size_t)i * N + n] = p[(size_t)n * elempack + i];
            }
        }
    }
}

static int conv1x1s1_apple_amx_fp16(const Mat& bottom_blob, Mat& top_blob, const Mat& _weight_fp16, const Mat& _bias, const Option& opt)
{
    const int N = (int)top_blob.w * top_blob.h;

    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const int M = (int)top_blob.c * out_elempack;
    const int K = (int)bottom_blob.c * elempack;

    // transposed fp16 weight, 32 output channels padded
    const __fp16* A = (const __fp16*)_weight_fp16;
    const int Mpad = (M + 31) / 32 * 32;
    const int lda = Mpad;

    const float* bias = _bias.empty() ? 0 : (const float*)_bias;

    Mat Bpack;
    const __fp16* B = 0;
    int ldb = 0;

    if (elempack == 1)
    {
        // pack1 input is already N fastest
        B = (const __fp16*)bottom_blob;
        ldb = (int)(bottom_blob.cstep * elempack);
    }
    else
    {
        conv1x1s1_apple_amx_pack_f16(bottom_blob, Bpack, K, N, opt);
        if (Bpack.empty())
            return -100;

        B = (const __fp16*)Bpack;
        ldb = N;
    }

    // cstep counts packed elements, convert to fp16 element offset
    const size_t out_cstep = top_blob.cstep * out_elempack;
    __fp16* outptr = (__fp16*)top_blob;

    // narrower than one 32 pixel tile is left to the native path by the caller
    const int mtiles = Mpad / 32;
    const int ntiles = (N + 31) / 32;
    const int total = mtiles * ntiles;

    // amx is enabled once per thread, set traps when it is already enabled and saving
    // the tile state for every 32x32 tile costs more than the tile itself
    #pragma omp parallel num_threads(opt.num_threads)
    {
        NCNN_AMX_SET();

        #pragma omp for schedule(static)
        for (int t = 0; t < total; t++)
        {
            const int mi = t / ntiles;
            const int ni = t % ntiles;

            const int m0 = mi * 32;
            // the last tile overlaps backwards so that the x operand never reads past N
            const int n0 = ni * 32 + 32 <= N ? ni * 32 : N - 32;
            // pixels below nkeep belong to the previous tile, storing them here would
            // let two workers write the same output element
            const int nkeep = ni * 32;
            const int d = nkeep - n0;

            float zt[64 * 16] __attribute__((aligned(64)));
            __fp16 t16[32 * 32] __attribute__((aligned(64)));

            // the first k skips the z input, so the tile needs no zeroing from set
            conv1x1s1_apple_amx_tile_f16(A, lda, B, ldb, m0, n0, K, zt);

            if (d)
            {
                for (int j = 0; j < 32; j++)
                {
                    const float b = (bias && m0 + j < M) ? bias[m0 + j] : 0.f;
                    conv1x1s1_apple_amx_store_row_f16(zt + j * 2 * 16, zt + (j * 2 + 1) * 16, b, t16 + j * 32);
                }

                for (int j = 0; j < 32; j++)
                {
                    const int m = m0 + j;
                    if (m >= M)
                        break;

                    __fp16* dst = outptr + (size_t)(m / out_elempack) * out_cstep + (size_t)nkeep * out_elempack + (m % out_elempack);

                    for (int c = d; c < 32; c++)
                    {
                        dst[(size_t)(c - d) * out_elempack] = t16[j * 32 + c];
                    }
                }
            }
            else if (out_elempack == 1)
            {
                for (int j = 0; j < 32; j++)
                {
                    const int m = m0 + j;
                    if (m >= M)
                        break;

                    const float b = bias ? bias[m] : 0.f;
                    __fp16* dst = outptr + (size_t)m * out_cstep + n0;

                    conv1x1s1_apple_amx_store_row_f16(zt + j * 2 * 16, zt + (j * 2 + 1) * 16, b, dst);
                }
            }
            else
            {
                for (int j = 0; j < 32; j++)
                {
                    const float b = (bias && m0 + j < M) ? bias[m0 + j] : 0.f;
                    conv1x1s1_apple_amx_store_row_f16(zt + j * 2 * 16, zt + (j * 2 + 1) * 16, b, t16 + j * 32);
                }

                if (out_elempack == 8)
                {
                    for (int g = 0; g < 4; g++)
                    {
                        const int mg = m0 + g * 8;
                        if (mg >= M)
                            break;

                        __fp16* dstc = outptr + (size_t)(mg / 8) * out_cstep + (size_t)n0 * 8;

                        for (int nb = 0; nb < 4; nb++)
                        {
                            conv1x1s1_apple_amx_transpose8x8_f16(t16 + (g * 8) * 32 + nb * 8, 32, dstc + nb * 8 * 8, 8);
                        }
                    }
                }
                else
                {
                    for (int g = 0; g < 8; g++)
                    {
                        const int mg = m0 + g * 4;
                        if (mg >= M)
                            break;

                        __fp16* dstc = outptr + (size_t)(mg / 4) * out_cstep + (size_t)n0 * 4;

                        for (int nb = 0; nb < 8; nb++)
                        {
                            conv1x1s1_apple_amx_transpose4x4_f16(t16 + (g * 4) * 32 + nb * 4, 32, dstc + nb * 4 * 4, 4);
                        }
                    }
                }
            }
        }

        NCNN_AMX_CLR();
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

// matint signed int8 x int8 with int32 accumulate
// alu mode 8 selects int8 multiplicands, lane width mode 10 selects the 64 row int32 z layout
// one instruction covers 16 output channels x 64 pixels, 1024 mac
// x = 64 consecutive int8 of input channel k, pixels n0 .. n0+63
// y = 64 bytes of which only byte 4m is read, so weight channels are stored stretched by 4
// z physical row 4m+s lane c accumulates into C[m0+m][n0+4c+s]
#define NCNN_AMX_MATINT_INT8_OPERAND ((1ull << 63) | (8ull << 47) | (10ull << 42) | (1ull << 26))

// set enables amx and zeroes x y z, clr disables amx again
// set raises an invalid instruction exception when amx is already enabled, so every tile is bracketed by set and clr
// there is no skip z flag in matint, the accumulator is zeroed by set instead
static inline void conv1x1s1_apple_amx_tile_int8(const signed char* A, const signed char* B, int ldb, int m0, int n0, int K, int* zt)
{
    const uint64_t op = NCNN_AMX_MATINT_INT8_OPERAND;
    const signed char* w = A + (size_t)(m0 / 16) * (size_t)K * 64;

    NCNN_AMX_SET();

    for (int k = 0; k < K; k++)
    {
        NCNN_AMX_LDX((uint64_t)(B + (size_t)k * ldb + n0));
        NCNN_AMX_LDY((uint64_t)(w + (size_t)k * 64));
        NCNN_AMX_MATINT(op);
    }

    for (int r = 0; r < 64; r++)
    {
        NCNN_AMX_STZ((uint64_t)(zt + r * 16) + ((uint64_t)r << 56));
    }

    NCNN_AMX_CLR();
}

// kxk stride1 int8 tile, the input plane of every channel is packed once and each kernel offset is a plain byte shift of it
// a column index n is the padded layout index oy * bw + ox, so the 64 byte x operand stays contiguous for every tap
// the weight rows must follow the same order, channel major and kernel offset minor
static inline void convks1_apple_amx_tile_int8(const signed char* A, const signed char* B, int ldb, int bw, int channels, int kernel_w, int kernel_h, int m0, int n0, int K, int* zt)
{
    const uint64_t op = NCNN_AMX_MATINT_INT8_OPERAND;
    const signed char* w = A + (size_t)(m0 / 16) * (size_t)K * 64;

    NCNN_AMX_SET();

    const signed char* a = w;

    for (int q = 0; q < channels; q++)
    {
        const signed char* bq = B + (size_t)q * ldb + n0;

        for (int ky = 0; ky < kernel_h; ky++)
        {
            const signed char* bky = bq + (size_t)ky * bw;

            for (int kx = 0; kx < kernel_w; kx++)
            {
                NCNN_AMX_LDX((uint64_t)(bky + kx));
                NCNN_AMX_LDY((uint64_t)a);
                NCNN_AMX_MATINT(op);
                a += 64;
            }
        }
    }

    for (int r = 0; r < 64; r++)
    {
        NCNN_AMX_STZ((uint64_t)(zt + r * 16) + ((uint64_t)r << 56));
    }

    NCNN_AMX_CLR();
}

// requantize one z row of 16 int32 into 16 int8, same math as requantize_from_int32_to_int8
static inline int8x16_t conv1x1s1_apple_amx_requant_int8(const int* zr, float scale_in, float bias, float scale_out, int activation_type, const Mat& activation_params)
{
    float32x4_t _scale_in = vdupq_n_f32(scale_in);
    float32x4_t _bias = vdupq_n_f32(bias);
    float32x4_t _scale_out = vdupq_n_f32(scale_out);

    float32x4_t v0 = vcvtq_f32_s32(vld1q_s32(zr));
    float32x4_t v1 = vcvtq_f32_s32(vld1q_s32(zr + 4));
    float32x4_t v2 = vcvtq_f32_s32(vld1q_s32(zr + 8));
    float32x4_t v3 = vcvtq_f32_s32(vld1q_s32(zr + 12));

    v0 = vfmaq_f32(_bias, v0, _scale_in);
    v1 = vfmaq_f32(_bias, v1, _scale_in);
    v2 = vfmaq_f32(_bias, v2, _scale_in);
    v3 = vfmaq_f32(_bias, v3, _scale_in);

    v0 = activation_ps(v0, activation_type, activation_params);
    v1 = activation_ps(v1, activation_type, activation_params);
    v2 = activation_ps(v2, activation_type, activation_params);
    v3 = activation_ps(v3, activation_type, activation_params);

    v0 = vmulq_f32(v0, _scale_out);
    v1 = vmulq_f32(v1, _scale_out);
    v2 = vmulq_f32(v2, _scale_out);
    v3 = vmulq_f32(v3, _scale_out);

    return vcombine_s8(float2int8(v0, v1), float2int8(v2, v3));
}

// transpose an 8x8 int8 block, row i of the source is src + i * src_stride
static inline void conv1x1s1_apple_amx_transpose8x8_i8(const signed char* src, int src_stride, signed char* dst, size_t dst_cstep)
{
    int8x8_t r0 = vld1_s8(src);
    int8x8_t r1 = vld1_s8(src + src_stride);
    int8x8_t r2 = vld1_s8(src + src_stride * 2);
    int8x8_t r3 = vld1_s8(src + src_stride * 3);
    int8x8_t r4 = vld1_s8(src + src_stride * 4);
    int8x8_t r5 = vld1_s8(src + src_stride * 5);
    int8x8_t r6 = vld1_s8(src + src_stride * 6);
    int8x8_t r7 = vld1_s8(src + src_stride * 7);

    int8x8_t a0 = vzip1_s8(r0, r1);
    int8x8_t a1 = vzip2_s8(r0, r1);
    int8x8_t a2 = vzip1_s8(r2, r3);
    int8x8_t a3 = vzip2_s8(r2, r3);
    int8x8_t a4 = vzip1_s8(r4, r5);
    int8x8_t a5 = vzip2_s8(r4, r5);
    int8x8_t a6 = vzip1_s8(r6, r7);
    int8x8_t a7 = vzip2_s8(r6, r7);

    int8x8_t b0 = vreinterpret_s8_s16(vzip1_s16(vreinterpret_s16_s8(a0), vreinterpret_s16_s8(a2)));
    int8x8_t b1 = vreinterpret_s8_s16(vzip2_s16(vreinterpret_s16_s8(a0), vreinterpret_s16_s8(a2)));
    int8x8_t b2 = vreinterpret_s8_s16(vzip1_s16(vreinterpret_s16_s8(a1), vreinterpret_s16_s8(a3)));
    int8x8_t b3 = vreinterpret_s8_s16(vzip2_s16(vreinterpret_s16_s8(a1), vreinterpret_s16_s8(a3)));
    int8x8_t b4 = vreinterpret_s8_s16(vzip1_s16(vreinterpret_s16_s8(a4), vreinterpret_s16_s8(a6)));
    int8x8_t b5 = vreinterpret_s8_s16(vzip2_s16(vreinterpret_s16_s8(a4), vreinterpret_s16_s8(a6)));
    int8x8_t b6 = vreinterpret_s8_s16(vzip1_s16(vreinterpret_s16_s8(a5), vreinterpret_s16_s8(a7)));
    int8x8_t b7 = vreinterpret_s8_s16(vzip2_s16(vreinterpret_s16_s8(a5), vreinterpret_s16_s8(a7)));

    int8x8_t c0 = vreinterpret_s8_s32(vzip1_s32(vreinterpret_s32_s8(b0), vreinterpret_s32_s8(b4)));
    int8x8_t c1 = vreinterpret_s8_s32(vzip2_s32(vreinterpret_s32_s8(b0), vreinterpret_s32_s8(b4)));
    int8x8_t c2 = vreinterpret_s8_s32(vzip1_s32(vreinterpret_s32_s8(b1), vreinterpret_s32_s8(b5)));
    int8x8_t c3 = vreinterpret_s8_s32(vzip2_s32(vreinterpret_s32_s8(b1), vreinterpret_s32_s8(b5)));
    int8x8_t c4 = vreinterpret_s8_s32(vzip1_s32(vreinterpret_s32_s8(b2), vreinterpret_s32_s8(b6)));
    int8x8_t c5 = vreinterpret_s8_s32(vzip2_s32(vreinterpret_s32_s8(b2), vreinterpret_s32_s8(b6)));
    int8x8_t c6 = vreinterpret_s8_s32(vzip1_s32(vreinterpret_s32_s8(b3), vreinterpret_s32_s8(b7)));
    int8x8_t c7 = vreinterpret_s8_s32(vzip2_s32(vreinterpret_s32_s8(b3), vreinterpret_s32_s8(b7)));

    vst1_s8(dst, c0);
    vst1_s8(dst + dst_cstep, c1);
    vst1_s8(dst + dst_cstep * 2, c2);
    vst1_s8(dst + dst_cstep * 3, c3);
    vst1_s8(dst + dst_cstep * 4, c4);
    vst1_s8(dst + dst_cstep * 5, c5);
    vst1_s8(dst + dst_cstep * 6, c6);
    vst1_s8(dst + dst_cstep * 7, c7);
}

// repack the int8 input as a K x N matrix with N contiguous
// slack reserves extra bytes past the last row for the kxk path, where a shifted 64 byte x operand may read past the plane
static void conv1x1s1_apple_amx_pack_int8(const Mat& bottom_blob, Mat& Bpack, int K, int N, int slack, const Option& opt)
{
    const int elempack = bottom_blob.elempack;

    Bpack.create((size_t)K * N + slack, (size_t)1u, opt.workspace_allocator);
    if (Bpack.empty())
        return;

    signed char* bp = (signed char*)Bpack;

    if (slack)
        memset(bp + (size_t)K * N, 0, slack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < bottom_blob.c; q++)
    {
        const signed char* p = (const signed char*)bottom_blob.channel(q);
        signed char* dst0 = bp + (size_t)(q * elempack) * N;

        if (elempack == 8)
        {
            int n = 0;
            for (; n + 8 <= N; n += 8)
            {
                conv1x1s1_apple_amx_transpose8x8_i8(p + (size_t)n * 8, 8, dst0 + n, N);
            }
            for (; n < N; n++)
            {
                for (int i = 0; i < 8; i++)
                    dst0[(size_t)i * N + n] = p[(size_t)n * 8 + i];
            }
        }
        else
        {
            for (int n = 0; n < N; n++)
            {
                for (int i = 0; i < elempack; i++)
                    dst0[(size_t)i * N + n] = p[(size_t)n * elempack + i];
            }
        }
    }
}

// requantize the 16 int32 channels of one tile into mlast rows of 64 consecutive int8 pixels
static inline void conv1x1s1_apple_amx_requant_tile_int8(const int* zt, int mlast, const float* scale_in_data, const float* bias_data, float scale_out, int m0, int activation_type, const Mat& activation_params, signed char* t)
{
    for (int m = 0; m < mlast; m++)
    {
        const int mc = m0 + m;
        const float s = scale_in_data[mc];
        const float b = bias_data ? bias_data[mc] : 0.f;

        const int* zr = zt + 4 * m * 16;

        int8x16_t r0 = conv1x1s1_apple_amx_requant_int8(zr, s, b, scale_out, activation_type, activation_params);
        int8x16_t r1 = conv1x1s1_apple_amx_requant_int8(zr + 16, s, b, scale_out, activation_type, activation_params);
        int8x16_t r2 = conv1x1s1_apple_amx_requant_int8(zr + 32, s, b, scale_out, activation_type, activation_params);
        int8x16_t r3 = conv1x1s1_apple_amx_requant_int8(zr + 48, s, b, scale_out, activation_type, activation_params);

        // interleave the four pixel phases back into 64 consecutive pixels
        int8x16x4_t v;
        v.val[0] = r0;
        v.val[1] = r1;
        v.val[2] = r2;
        v.val[3] = r3;
        vst4q_s8(t + m * 64, v);
    }
}

static int conv1x1s1_apple_amx_int8(const Mat& bottom_blob, Mat& top_blob, const Mat& _weight_amx, const Mat& _scale_in_data, const Mat& _bias_data, const Mat& _scale_out_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int N = (int)top_blob.w * top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int M = (int)top_blob.c * out_elempack;

    const int elempack = bottom_blob.elempack;
    const int K = (int)bottom_blob.c * elempack;

    const signed char* A = (const signed char*)_weight_amx;
    const float* scale_in_data = _scale_in_data;
    const float* bias_data = _bias_data.empty() ? 0 : (const float*)_bias_data;
    const float scale_out = _scale_out_data.empty() ? 1.f : ((const float*)_scale_out_data)[0];

    Mat Bpack;
    const signed char* B = 0;
    int ldb = 0;

    if (elempack == 1)
    {
        // pack1 input is already N fastest
        B = (const signed char*)bottom_blob;
        ldb = (int)bottom_blob.cstep;
    }
    else
    {
        conv1x1s1_apple_amx_pack_int8(bottom_blob, Bpack, K, N, 0, opt);
        if (Bpack.empty())
            return -100;

        B = (const signed char*)Bpack;
        ldb = N;
    }

    // cstep counts packed elements, convert to byte offset
    const size_t out_cstep = top_blob.cstep * out_elempack;
    signed char* outptr = (signed char*)top_blob;

    const int mtiles = (M + 15) / 16;
    const int ntiles = (N + 63) / 64;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int t = 0; t < mtiles * ntiles; t++)
    {
        const int mi = t / ntiles;
        const int ni = t % ntiles;

        const int m0 = mi * 16;
        // the last tile overlaps backwards so that the x operand never reads past N
        const int n0 = ni * 64 + 64 <= N ? ni * 64 : N - 64;
        // columns below nkeep are owned by the previous tile, storing them here would let
        // two workers write the same output element
        const int nkeep = ni * 64;
        const int d = nkeep - n0;

        int zt[64 * 16] __attribute__((aligned(64)));
        signed char t8[16 * 64] __attribute__((aligned(64)));

        conv1x1s1_apple_amx_tile_int8(A, B, ldb, m0, n0, K, zt);

        const int mlast = M - m0 < 16 ? M - m0 : 16;

        conv1x1s1_apple_amx_requant_tile_int8(zt, mlast, scale_in_data, bias_data, scale_out, m0, activation_type, activation_params, t8);

        if (out_elempack == 1)
        {
            for (int m = 0; m < mlast; m++)
            {
                signed char* dst = outptr + (size_t)(m0 + m) * out_cstep + nkeep;

                memcpy(dst, t8 + m * 64 + d, 64 - d);
            }
        }
        else
        {
            // transpose 8 output channels at a time into the packed pixel layout
            for (int g = 0; g < mlast / 8; g++)
            {
                signed char* dstc = outptr + (size_t)((m0 + g * 8) / 8) * out_cstep + (size_t)nkeep * 8;

                for (int nb = 0; nb < 8; nb++)
                {
                    const int c0 = n0 + nb * 8;
                    if (c0 + 8 <= nkeep)
                        continue;

                    if (c0 >= nkeep)
                    {
                        conv1x1s1_apple_amx_transpose8x8_i8(t8 + (g * 8) * 64 + nb * 8, 64, dstc + (size_t)(c0 - nkeep) * 8, 8);
                    }
                    else
                    {
                        // the tile boundary cuts this block, store its tail only
                        for (int j = nkeep - c0; j < 8; j++)
                        {
                            for (int i = 0; i < 8; i++)
                            {
                                dstc[(size_t)(c0 + j - nkeep) * 8 + i] = t8[(g * 8 + i) * 64 + nb * 8 + j];
                            }
                        }
                    }
                }
            }
        }
    }

    return 0;
}

// kxk stride1 int8 convolution through the same matint kernel
// the input is packed once as channels x (bh * bw) and every tap is a byte shift, so no im2col buffer is materialized
// the epilogue maps the padded column layout back to the real output rows while it writes
static int convks1_apple_amx_int8(const Mat& bottom_blob, Mat& top_blob, const Mat& _weight_amx, const Mat& _scale_in_data, const Mat& _bias_data, const Mat& _scale_out_data, int kernel_w, int kernel_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int outw = (int)top_blob.w;
    const int outh = (int)top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int M = (int)top_blob.c * out_elempack;

    const int bw = (int)bottom_blob.w;
    const int Np = bw * (int)bottom_blob.h;
    // columns of the padded layout, the bw - outw tails of each row are computed and dropped
    const int Ntotal = bw * outh;

    const int elempack = bottom_blob.elempack;
    const int channels = (int)bottom_blob.c * elempack;
    const int K = channels * kernel_w * kernel_h;

    const signed char* A = (const signed char*)_weight_amx;
    const float* scale_in_data = _scale_in_data;
    const float* bias_data = _bias_data.empty() ? 0 : (const float*)_bias_data;
    const float scale_out = _scale_out_data.empty() ? 1.f : ((const float*)_scale_out_data)[0];

    // the last tap shifts the x operand by kernel_w - 1 bytes past the end of the plane and
    // reads a whole 64 byte operand from there, so the slack has to follow the kernel width.
    // those bytes only cover the dropped row tail columns, zero or foreign data is never stored
    const int bslack = kernel_w > 64 ? kernel_w : 64;
    Mat Bpack;
    conv1x1s1_apple_amx_pack_int8(bottom_blob, Bpack, channels, Np, bslack, opt);
    if (Bpack.empty())
        return -100;

    const signed char* B = (const signed char*)Bpack;

    // cstep counts packed elements, convert to byte offset
    const size_t out_cstep = top_blob.cstep * out_elempack;
    signed char* outptr = (signed char*)top_blob;

    const int mtiles = (M + 15) / 16;
    const int ntiles = (Ntotal + 63) / 64;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int t = 0; t < mtiles * ntiles; t++)
    {
        const int mi = t / ntiles;
        const int ni = t % ntiles;

        const int m0 = mi * 16;
        // the last tile overlaps backwards so that the x operand never reads past the padded plane
        const int n0 = ni * 64 + 64 <= Ntotal ? ni * 64 : Ntotal - 64;
        // padded columns below nkeep were already computed by the previous tile, storing
        // them again would let two workers write the same output pixel
        const int nkeep = ni * 64;

        int zt[64 * 16] __attribute__((aligned(64)));
        signed char t8[16 * 64] __attribute__((aligned(64)));

        convks1_apple_amx_tile_int8(A, B, Np, bw, channels, kernel_w, kernel_h, m0, n0, K, zt);

        const int mlast = M - m0 < 16 ? M - m0 : 16;

        conv1x1s1_apple_amx_requant_tile_int8(zt, mlast, scale_in_data, bias_data, scale_out, m0, activation_type, activation_params, t8);

        // every 8 padded columns either land in one output row or straddle the dropped row tail
        for (int nb = 0; nb < 8; nb++)
        {
            const int c0 = n0 + nb * 8;
            if (c0 + 8 <= nkeep)
                continue;

            const int oy = c0 / bw;
            const int ox = c0 % bw;

            if (ox + 8 <= outw && c0 >= nkeep)
            {
                const size_t outbase = (size_t)oy * outw + ox;

                if (out_elempack == 1)
                {
                    for (int m = 0; m < mlast; m++)
                    {
                        signed char* dst = outptr + (size_t)(m0 + m) * out_cstep + outbase;

                        memcpy(dst, t8 + m * 64 + nb * 8, 8);
                    }
                }
                else
                {
                    for (int g = 0; g < mlast / 8; g++)
                    {
                        signed char* dstc = outptr + (size_t)((m0 + g * 8) / 8) * out_cstep + outbase * 8;

                        conv1x1s1_apple_amx_transpose8x8_i8(t8 + (g * 8) * 64 + nb * 8, 64, dstc, 8);
                    }
                }
            }
            else
            {
                for (int j = 0; j < 8; j++)
                {
                    const int c = c0 + j;
                    if (c < nkeep)
                        continue;

                    const int cy = c / bw;
                    const int cx = c % bw;
                    if (cx >= outw)
                        continue;

                    const size_t outbase = (size_t)cy * outw + cx;

                    if (out_elempack == 1)
                    {
                        for (int m = 0; m < mlast; m++)
                        {
                            outptr[(size_t)(m0 + m) * out_cstep + outbase] = t8[m * 64 + nb * 8 + j];
                        }
                    }
                    else
                    {
                        for (int g = 0; g < mlast / 8; g++)
                        {
                            signed char* dstc = outptr + (size_t)((m0 + g * 8) / 8) * out_cstep + outbase * 8;

                            for (int i = 0; i < 8; i++)
                            {
                                dstc[i] = t8[(g * 8 + i) * 64 + nb * 8 + j];
                            }
                        }
                    }
                }
            }
        }
    }

    return 0;
}

#endif // defined __APPLE__ && __aarch64__
