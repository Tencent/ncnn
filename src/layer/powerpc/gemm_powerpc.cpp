// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gemm_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

#if __PPU__
#include "gemm_powerpc_spu.h"
#endif // __PPU__

namespace ncnn {

Gemm_powerpc::Gemm_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

#if __ALTIVEC__
// plain fp32 C = alpha * A * B^T + beta * C with scalar broadcast C
// A is [M x K] with elempack 4, i.e. A(i,k) = ((float*)A)[((i/4)*K + k) * 4 + i%4]
// BT is [N x K] with elempack 4, i.e. BT(j,k) = ((float*)BT)[((j/4)*K + k) * 4 + j%4]
// top is [N x M/4] with elempack 4, i.e. top(i,j) = ((float*)top)[((i/4)*N + j) * 4 + i%4]
static int gemm_transB_pack4_powerpc(const Mat& A, const Mat& BT, Mat& top_blob, float alpha, float bc0, const Option& opt)
{
    const int M = A.h * 4;
    const int N = BT.h * 4;
    const int K = A.w;

    const v4f32 _zero = vec_splats(0.f);
    const v4f32 _alpha = vec_splats(alpha);
    const v4f32 _bias = vec_splats(bc0);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < A.h; q++)
    {
        const float* aptr = (const float*)A + (size_t)q * K * 4;
        float* outptr = (float*)top_blob + (size_t)q * N * 4;

        for (int jg = 0; jg < BT.h; jg++)
        {
            const float* bptr = (const float*)BT + (size_t)jg * K * 4;

            v4f32 _s0 = _bias;
            v4f32 _s1 = _bias;
            v4f32 _s2 = _bias;
            v4f32 _s3 = _bias;

            for (int k = 0; k < K; k++)
            {
                v4f32 _a = __ncnn_altivec_load(aptr + k * 4);
                v4f32 _b = __ncnn_altivec_load(bptr + k * 4);

                _s0 = vec_madd(_a, vec_splat(_b, 0), _s0);
                _s1 = vec_madd(_a, vec_splat(_b, 1), _s1);
                _s2 = vec_madd(_a, vec_splat(_b, 2), _s2);
                _s3 = vec_madd(_a, vec_splat(_b, 3), _s3);
            }

            __ncnn_altivec_store(outptr, vec_madd(_s0, _alpha, _zero));
            __ncnn_altivec_store(outptr + 4, vec_madd(_s1, _alpha, _zero));
            __ncnn_altivec_store(outptr + 8, vec_madd(_s2, _alpha, _zero));
            __ncnn_altivec_store(outptr + 12, vec_madd(_s3, _alpha, _zero));
            outptr += 16;
        }
    }

    return 0;
}

// plain fp32 C = alpha * A * B^T + beta * C with scalar broadcast C
// A and BT are plain row-major [M x K] and [N x K], top is [N x M] with elempack 1 or 4
static int gemm_transB_pack1_powerpc(const Mat& A, const Mat& BT, Mat& top_blob, float alpha, float bc0, const Option& opt)
{
    const int M = A.dims == 3 ? A.c : A.h;
    const int N = BT.dims == 3 ? BT.c : BT.h;
    const int K = A.w;

    const size_t astride = A.dims == 3 ? A.cstep : (size_t)A.w;
    const size_t bstride = BT.dims == 3 ? BT.cstep : (size_t)BT.w;

    const int out4 = top_blob.elempack == 4;
    float* out = (float*)top_blob;

    // 4x4 micro-tile over full row groups, remainder rows are handled below
    const int M4 = M / 4 * 4;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < M4 / 4; q++)
    {
        const int i = q * 4;
        const float* a0 = (const float*)A + i * astride;
        const float* a1 = a0 + astride;
        const float* a2 = a0 + astride * 2;
        const float* a3 = a0 + astride * 3;

        float* outrow = out + (out4 ? (size_t)q * N * 4 : (size_t)i * N);

        int j = 0;
        for (; j + 3 < N; j += 4)
        {
            const float* b0 = (const float*)BT + j * bstride;
            const float* b1 = b0 + bstride;
            const float* b2 = b0 + bstride * 2;
            const float* b3 = b0 + bstride * 3;

            v4f32 _s[4][4];
            for (int l = 0; l < 4; l++)
            {
                for (int t = 0; t < 4; t++)
                {
                    _s[l][t] = vec_splats(0.f);
                }
            }

            int k = 0;
            for (; k + 3 < K; k += 4)
            {
                v4f32 _a0 = __ncnn_altivec_load(a0 + k);
                v4f32 _a1 = __ncnn_altivec_load(a1 + k);
                v4f32 _a2 = __ncnn_altivec_load(a2 + k);
                v4f32 _a3 = __ncnn_altivec_load(a3 + k);

                v4f32 _b0 = __ncnn_altivec_load(b0 + k);
                v4f32 _b1 = __ncnn_altivec_load(b1 + k);
                v4f32 _b2 = __ncnn_altivec_load(b2 + k);
                v4f32 _b3 = __ncnn_altivec_load(b3 + k);

                _s[0][0] = vec_madd(_a0, _b0, _s[0][0]);
                _s[0][1] = vec_madd(_a0, _b1, _s[0][1]);
                _s[0][2] = vec_madd(_a0, _b2, _s[0][2]);
                _s[0][3] = vec_madd(_a0, _b3, _s[0][3]);

                _s[1][0] = vec_madd(_a1, _b0, _s[1][0]);
                _s[1][1] = vec_madd(_a1, _b1, _s[1][1]);
                _s[1][2] = vec_madd(_a1, _b2, _s[1][2]);
                _s[1][3] = vec_madd(_a1, _b3, _s[1][3]);

                _s[2][0] = vec_madd(_a2, _b0, _s[2][0]);
                _s[2][1] = vec_madd(_a2, _b1, _s[2][1]);
                _s[2][2] = vec_madd(_a2, _b2, _s[2][2]);
                _s[2][3] = vec_madd(_a2, _b3, _s[2][3]);

                _s[3][0] = vec_madd(_a3, _b0, _s[3][0]);
                _s[3][1] = vec_madd(_a3, _b1, _s[3][1]);
                _s[3][2] = vec_madd(_a3, _b2, _s[3][2]);
                _s[3][3] = vec_madd(_a3, _b3, _s[3][3]);
            }

            float sum[4][4];
            for (int l = 0; l < 4; l++)
            {
                for (int t = 0; t < 4; t++)
                {
                    sum[l][t] = __ncnn_altivec_reduce_add(_s[l][t]);
                }
            }

            for (; k < K; k++)
            {
                sum[0][0] += a0[k] * b0[k];
                sum[0][1] += a0[k] * b1[k];
                sum[0][2] += a0[k] * b2[k];
                sum[0][3] += a0[k] * b3[k];
                sum[1][0] += a1[k] * b0[k];
                sum[1][1] += a1[k] * b1[k];
                sum[1][2] += a1[k] * b2[k];
                sum[1][3] += a1[k] * b3[k];
                sum[2][0] += a2[k] * b0[k];
                sum[2][1] += a2[k] * b1[k];
                sum[2][2] += a2[k] * b2[k];
                sum[2][3] += a2[k] * b3[k];
                sum[3][0] += a3[k] * b0[k];
                sum[3][1] += a3[k] * b1[k];
                sum[3][2] += a3[k] * b2[k];
                sum[3][3] += a3[k] * b3[k];
            }

            if (out4)
            {
                for (int t = 0; t < 4; t++)
                {
                    float v[4];
                    v[0] = (sum[0][t] + bc0) * alpha;
                    v[1] = (sum[1][t] + bc0) * alpha;
                    v[2] = (sum[2][t] + bc0) * alpha;
                    v[3] = (sum[3][t] + bc0) * alpha;
                    __ncnn_altivec_store(outrow + t * 4, __ncnn_altivec_load(v));
                }
            }
            else
            {
                for (int l = 0; l < 4; l++)
                {
                    for (int t = 0; t < 4; t++)
                    {
                        outrow[l * N + j + t] = (sum[l][t] + bc0) * alpha;
                    }
                }
            }

            if (out4)
                outrow += 16;
        }

        // remainder columns
        for (; j < N; j++)
        {
            const float* b = (const float*)BT + j * bstride;

            float sum[4] = {0.f, 0.f, 0.f, 0.f};
            for (int k = 0; k < K; k++)
            {
                const float bval = b[k];
                sum[0] += a0[k] * bval;
                sum[1] += a1[k] * bval;
                sum[2] += a2[k] * bval;
                sum[3] += a3[k] * bval;
            }

            if (out4)
            {
                float v[4];
                v[0] = (sum[0] + bc0) * alpha;
                v[1] = (sum[1] + bc0) * alpha;
                v[2] = (sum[2] + bc0) * alpha;
                v[3] = (sum[3] + bc0) * alpha;
                __ncnn_altivec_store(outrow + j * 4, __ncnn_altivec_load(v));
            }
            else
            {
                for (int l = 0; l < 4; l++)
                {
                    outrow[l * N + j] = (sum[l] + bc0) * alpha;
                }
            }
        }
    }

    // remainder rows
    if (M4 < M)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = M4; i < M; i++)
        {
            const float* a = (const float*)A + i * astride;
            float* outrow = out + (size_t)i * N;

            for (int j = 0; j < N; j++)
            {
                const float* b = (const float*)BT + j * bstride;

                float sum = 0.f;
                for (int k = 0; k < K; k++)
                {
                    sum += a[k] * b[k];
                }

                outrow[j] = (sum + bc0) * alpha;
            }
        }
    }

    return 0;
}
#endif // __ALTIVEC__

int Gemm_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if __ALTIVEC__
    if (!weight_block_quantize && !quantize_term && !constantA && !constantB
            && (!constantC || constant_broadcast_type_C == -1 || constant_broadcast_type_C == 0)
            && transA == 0 && transB == 1
            && (beta == 0.f || beta == 1.f)
            && !output_transpose && !output_N1M && !output_elempack && output_elemtype == 0
            && bottom_blobs.size() >= 2 && bottom_blobs[0].elembits() == 32
            && bottom_blobs[0].elempack == bottom_blobs[1].elempack
            && (bottom_blobs[0].elempack == 1 || bottom_blobs[0].elempack == 4)
            && bottom_blobs[0].w == bottom_blobs[1].w
            && (bottom_blobs[0].dims == 2 || bottom_blobs[0].dims == 3)
            && (bottom_blobs[1].dims == 2 || bottom_blobs[1].dims == 3)
            && (bottom_blobs[0].elempack == 1 || (bottom_blobs[0].dims == 2 && bottom_blobs[1].dims == 2)))
    {
        const Mat& A = bottom_blobs[0];
        const Mat& B = bottom_blobs[1];

        const int M = (A.dims == 3 ? A.c : A.h) * A.elempack;
        const int N = (B.dims == 3 ? B.c : B.h) * B.elempack;

        // only the scalar broadcast of C is handled here
        Mat C;
        if (constantC && constant_broadcast_type_C == 0)
        {
            C = C_data;
        }
        else if (bottom_blobs.size() >= 3)
        {
            C = bottom_blobs[2];
        }

        int broadcast_type_C = 0;
        if (!C.empty())
        {
            if (C.dims == 1 && C.w * C.elempack == 1)
                broadcast_type_C = 0;
            if (C.dims == 1 && C.w * C.elempack == M)
                broadcast_type_C = 1;
            if (C.dims == 1 && C.w * C.elempack == N)
                broadcast_type_C = 4;
            if (C.dims == 2 && C.w == 1 && C.h * C.elempack == M)
                broadcast_type_C = 2;
            if (C.dims == 2 && C.w == N && C.h * C.elempack == M)
                broadcast_type_C = 3;
            if (C.dims == 2 && C.w == N && C.h * C.elempack == 1)
                broadcast_type_C = 4;
        }

        if (broadcast_type_C == 0)
        {
            const float bc0 = (!C.empty() && beta == 1.f) ? C[0] : 0.f;

            const int out_elempack = (A.elempack == 4 || (M % 4 == 0 && opt.use_packing_layout)) ? 4 : 1;
            const size_t out_elemsize = 4u * out_elempack;

            Mat& top_blob = top_blobs[0];
            if (out_elempack == 4)
                top_blob.create(N, M / 4, out_elemsize, 4, opt.blob_allocator);
            else
                top_blob.create(N, M, out_elemsize, 1, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (A.elempack == 4)
            {
#if __PPU__
                // the spes handle the aligned pack4 case; anything else, or a
                // spe that fails to start, falls through to the ppe kernel
                if (gemm_transB_pack4_spu(A, B, top_blob, alpha, beta, bc0, 0) == 0)
                    return 0;
#endif // __PPU__
                return gemm_transB_pack4_powerpc(A, B, top_blob, alpha, bc0, opt);
            }

            return gemm_transB_pack1_powerpc(A, B, top_blob, alpha, bc0, opt);
        }
    }
#endif // __ALTIVEC__

    // generic fallback, the generic kernel is not packing aware
    std::vector<Mat> bottom_blobs_unpacked;
    bottom_blobs_unpacked.resize(bottom_blobs.size());
    for (size_t b = 0; b < bottom_blobs.size(); b++)
    {
        const Mat& blob = bottom_blobs[b];
        if (blob.empty() || blob.elempack == 1)
        {
            bottom_blobs_unpacked[b] = blob;
            continue;
        }

        convert_packing(blob, bottom_blobs_unpacked[b], 1, opt);
        if (bottom_blobs_unpacked[b].empty())
            return -100;
    }

    int ret = Gemm::forward(bottom_blobs_unpacked, top_blobs, opt);
    if (ret != 0)
        return ret;

#if __ALTIVEC__
    // repack the output when possible
    if (opt.use_packing_layout && !output_transpose && !output_elempack
            && top_blobs[0].dims == 2 && top_blobs[0].elempack == 1 && top_blobs[0].h % 4 == 0)
    {
        Mat top_blob_packed;
        convert_packing(top_blobs[0], top_blob_packed, 4, opt);
        // h is divisible by 4, so an empty result is an allocation failure,
        // not a shape convert_packing refused. falling through would hide the oom.
        if (top_blob_packed.empty())
            return -100;
        top_blobs[0] = top_blob_packed;
    }
#endif // __ALTIVEC__

    return ret;
}

} // namespace ncnn
