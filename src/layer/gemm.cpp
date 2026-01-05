// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gemm.h"

namespace ncnn {

Gemm::Gemm()
{
    one_blob_only = false;
    support_inplace = false;
}

int Gemm::load_param(const ParamDict& pd)
{
    alpha = pd.get(0, 1.f);
    beta = pd.get(1, 1.f);
    transA = pd.get(2, 0);
    transB = pd.get(3, 0);
    constantA = pd.get(4, 0);
    constantB = pd.get(5, 0);
    constantC = pd.get(6, 0);
    constantM = pd.get(7, 0);
    constantN = pd.get(8, 0);
    constantK = pd.get(9, 0);
    constant_broadcast_type_C = pd.get(10, 0);
    output_N1M = pd.get(11, 0);
    output_elempack = pd.get(12, 0);
    output_elemtype = pd.get(13, 0);
    output_transpose = pd.get(14, 0);
    int8_scale_term = pd.get(18, 0);
    constant_TILE_M = pd.get(20, 0);
    constant_TILE_N = pd.get(21, 0);
    constant_TILE_K = pd.get(22, 0);

    if (int8_scale_term)
    {
#if !NCNN_INT8
        NCNN_LOGE("please build ncnn with NCNN_INT8 enabled for int8 inference");
        return -1;
#endif
    }

    if (constantA == 1 && (constantM == 0 || constantK == 0))
    {
        NCNN_LOGE("constantM and constantK must be non-zero when constantA enabled");
        return -1;
    }

    if (constantB == 1 && (constantN == 0 || constantK == 0))
    {
        NCNN_LOGE("constantN and constantK must be non-zero when constantB enabled");
        return -1;
    }

    if (constantC == 1 && (constant_broadcast_type_C < -1 || constant_broadcast_type_C > 4))
    {
        NCNN_LOGE("constant_broadcast_type_C must be -1 or 0~4 when constantC enabled");
        return -1;
    }

    if (constantA == 0 && constantB == 1 && constantC == 1)
        one_blob_only = true;

    if (constantA == 1 && constantB == 0 && constantC == 1)
        one_blob_only = true;

    if (constantA == 1 && constantB == 1 && constantC == 0)
        one_blob_only = true;

    return 0;
}

int Gemm::load_model(const ModelBin& mb)
{
    if (constantA == 1)
    {
        if (transA == 0)
            A_data = mb.load(constantK, constantM, 0);
        else
            A_data = mb.load(constantM, constantK, 0);
        if (A_data.empty())
            return -100;
    }

    if (constantB == 1)
    {
        if (transB == 0)
            B_data = mb.load(constantN, constantK, 0);
        else
            B_data = mb.load(constantK, constantN, 0);
        if (B_data.empty())
            return -100;
    }

    if (constantC == 1 && constant_broadcast_type_C != -1)
    {
        if (constant_broadcast_type_C == 0)
            C_data = mb.load(1, 0);
        if (constant_broadcast_type_C == 1)
            C_data = mb.load(constantM, 0);
        if (constant_broadcast_type_C == 2)
            C_data = mb.load(1, constantM, 0);
        if (constant_broadcast_type_C == 3)
            C_data = mb.load(constantN, constantM, 0);
        if (constant_broadcast_type_C == 4)
            C_data = mb.load(constantN, 1, 0);
        if (C_data.empty())
            return -100;
    }

#if NCNN_INT8
    if (int8_scale_term)
    {
        if (constantA == 1)
        {
            A_data_int8_scales = mb.load(constantM, 1);
        }

        if (constantB == 1)
        {
            B_data_int8_scale = mb.load(1, 1)[0];
        }
    }
#endif // NCNN_INT8

    return 0;
}

static void gemm_transB(const Mat& A, const Mat& BT, const Mat& C, Mat& top_blob, float alpha, float beta, int broadcast_type_C, int output_transpose, const Option& opt)
{
    const int M = A.dims == 3 ? A.c : A.h;
    const int N = BT.dims == 3 ? BT.c : BT.h;
    const int K = A.w; // assert A.w == BT.w

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < M; i++)
    {
        const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

        const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
        const size_t BT_hstep = BT.dims == 3 ? BT.cstep : (size_t)BT.w;

        const float* ptrA = (const float*)A + i * A_hstep;
        const float* ptrC = C;

        for (int j = 0; j < N; j++)
        {
            const float* ptrBT = (const float*)BT + j * BT_hstep;

            float sum = 0.f;
            if (ptrC)
            {
                if (broadcast_type_C == 0)
                {
                    sum = ptrC[0];
                }
                if (broadcast_type_C == 1)
                {
                    sum = ptrC[i];
                }
                if (broadcast_type_C == 2)
                {
                    sum = ptrC[i];
                }
                if (broadcast_type_C == 3)
                {
                    sum = ptrC[i * N + j];
                }
                if (broadcast_type_C == 4)
                {
                    sum = ptrC[j];
                }

                sum *= beta;
            }

            for (int k = 0; k < K; k++)
            {
                sum += ptrA[k] * ptrBT[k];
            }

            sum *= alpha;

            if (output_transpose)
            {
                top_blob[j * out_hstep + i] = sum;
            }
            else
            {
                top_blob[i * out_hstep + j] = sum;
            }
        }
    }
}

#if NCNN_INT8
static inline signed char float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

static void gemm_transB_int8(const Mat& A_int8, const Mat& BT_int8, const Mat& A_int8_scales, float BT_int8_scale, const Mat& C, Mat& top_blob, float alpha, float beta, int broadcast_type_C, int output_transpose, const Option& opt)
{
    const int M = A_int8.h;
    const int N = BT_int8.h;
    const int K = A_int8.w; // assert A_int8.w == BT_int8.w

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < M; i++)
    {
        const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

        const signed char* ptrA = A_int8.row<const signed char>(i);
        const float* ptrC = C;

        const float descale = 1.f / (A_int8_scales[i] * BT_int8_scale);

        for (int j = 0; j < N; j++)
        {
            const signed char* ptrBT = BT_int8.row<const signed char>(j);

            int sum = 0;
            for (int k = 0; k < K; k++)
            {
                sum += ptrA[k] * ptrBT[k];
            }

            float sum_fp32 = sum * descale;

            if (ptrC)
            {
                float c = 0.f;
                if (broadcast_type_C == 0)
                {
                    c = ptrC[0];
                }
                if (broadcast_type_C == 1)
                {
                    c = ptrC[i];
                }
                if (broadcast_type_C == 2)
                {
                    c = ptrC[i];
                }
                if (broadcast_type_C == 3)
                {
                    c = ptrC[i * N + j];
                }
                if (broadcast_type_C == 4)
                {
                    c = ptrC[j];
                }

                sum_fp32 += c * beta;
            }

            sum_fp32 *= alpha;

            if (output_transpose)
            {
                top_blob[j * out_hstep + i] = sum_fp32;
            }
            else
            {
                top_blob[i * out_hstep + j] = sum_fp32;
            }
        }
    }
}
#endif // NCNN_INT8

int Gemm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    std::vector<Mat> bottom_blobs(1, bottom_blob);
    std::vector<Mat> top_blobs(1, top_blob);
    int ret = forward(bottom_blobs, top_blobs, opt);
    top_blob = top_blobs[0];
    return ret;
}

int Gemm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        return forward_int8(bottom_blobs, top_blobs, opt);
    }
#endif // NCNN_INT8

    const Mat& A0 = constantA ? A_data : bottom_blobs[0];
    const Mat& B0 = constantB ? B_data : constantA ? bottom_blobs[0] : bottom_blobs[1];

    size_t elemsize = A0.elemsize;

    Mat A;
    if (transA == 0)
    {
        A = A0;
    }
    else
    {
        // transpose A to row-major
        A.create((A0.dims == 3 ? A0.c : A0.h), A0.w, elemsize, opt.workspace_allocator);
        if (A.empty())
            return -100;

        const size_t A0_hstep = A0.dims == 3 ? A0.cstep : (size_t)A0.w;

        for (int i = 0; i < A.h; i++)
        {
            float* ptr = A.row(i);
            for (int j = 0; j < A.w; j++)
            {
                ptr[j] = A0[j * A0_hstep + i];
            }
        }
    }

    Mat BT;
    if (transB == 0)
    {
        // transpose B to col-major
        BT.create((B0.dims == 3 ? B0.c : B0.h), B0.w, elemsize, opt.workspace_allocator);
        if (BT.empty())
            return -100;

        const size_t B0_hstep = B0.dims == 3 ? B0.cstep : (size_t)B0.w;

        for (int i = 0; i < BT.h; i++)
        {
            float* ptr = BT.row(i);
            for (int j = 0; j < BT.w; j++)
            {
                ptr[j] = B0[j * B0_hstep + i];
            }
        }
    }
    else
    {
        BT = B0;
    }

    const int M = A.dims == 3 ? A.c : A.h;
    const int N = BT.dims == 3 ? BT.c : BT.h;

    Mat C;
    int broadcast_type_C = 0;
    if (constantC)
    {
        C = C_data;
        broadcast_type_C = constant_broadcast_type_C;
    }
    else
    {
        if (constantA && constantB && bottom_blobs.size() == 1)
        {
            C = bottom_blobs[0];
        }
        else if ((constantA || constantB) && bottom_blobs.size() == 2)
        {
            C = bottom_blobs[1];
        }
        else if (bottom_blobs.size() == 3)
        {
            C = bottom_blobs[2];
        }

        if (!C.empty())
        {
            if (C.dims == 1 && C.w == 1)
            {
                // scalar
                broadcast_type_C = 0;
            }
            if (C.dims == 1 && C.w == M)
            {
                // M
                // auto broadcast from h to w is the ncnn-style convention
                broadcast_type_C = 1;
            }
            if (C.dims == 1 && C.w == N)
            {
                // N
                broadcast_type_C = 4;
            }
            if (C.dims == 2 && C.w == 1 && C.h == M)
            {
                // Mx1
                broadcast_type_C = 2;
            }
            if (C.dims == 2 && C.w == N && C.h == M)
            {
                // MxN
                broadcast_type_C = 3;
            }
            if (C.dims == 2 && C.w == N && C.h == 1)
            {
                // 1xN
                broadcast_type_C = 4;
            }
        }
    }

    Mat& top_blob = top_blobs[0];
    if (output_transpose)
    {
        if (output_N1M)
            top_blob.create(M, 1, N, elemsize, opt.blob_allocator);
        else
            top_blob.create(M, N, elemsize, opt.blob_allocator);
    }
    else
    {
        if (output_N1M)
            top_blob.create(N, 1, M, elemsize, opt.blob_allocator);
        else
            top_blob.create(N, M, elemsize, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    gemm_transB(A, BT, C, top_blob, alpha, beta, broadcast_type_C, output_transpose, opt);

    return 0;
}

#if NCNN_INT8
int Gemm::forward_int8(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& A0 = constantA ? A_data : bottom_blobs[0];
    const Mat& B0 = constantB ? B_data : constantA ? bottom_blobs[0] : bottom_blobs[1];

    Mat A;
    if (transA == 0)
    {
        A = A0;
    }
    else
    {
        // transpose A to row-major
        if (A0.elemsize == 1)
        {
            A.create(A0.h, A0.w, (size_t)1u, 1, opt.workspace_allocator);
            if (A.empty())
                return -100;

            for (int i = 0; i < A.h; i++)
            {
                signed char* ptr = A.row<signed char>(i);
                for (int j = 0; j < A.w; j++)
                {
                    ptr[j] = A0.row<const signed char>(j)[i];
                }
            }
        }
        else
        {
            A.create(A0.dims == 3 ? A0.c : A0.h, A0.w, (size_t)4u, 1, opt.workspace_allocator);
            if (A.empty())
                return -100;

            for (int i = 0; i < A.h; i++)
            {
                float* ptr = A.row(i);
                for (int j = 0; j < A.w; j++)
                {
                    ptr[j] = A0.dims == 3 ? A0.channel(j)[i] : A0.row(j)[i];
                }
            }
        }
    }

    // dynamic quantize A
    Mat A_int8 = A;
    Mat A_int8_scales = A_data_int8_scales;
    if (A_int8.elemsize != 1)
    {
        A_int8.create(A.w, A.dims == 3 ? A.c : A.h, (size_t)1u, 1, opt.workspace_allocator);
        if (A_int8.empty())
            return -100;
        A_int8_scales.create(A_int8.h, (size_t)4u, 1, opt.workspace_allocator);
        if (A_int8_scales.empty())
            return -100;

        for (int i = 0; i < A_int8.h; i++)
        {
            const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
            const float* ptr = (const float*)A + i * A_hstep;

            float absmax = 0.f;
            for (int k = 0; k < A_int8.w; k++)
            {
                absmax = std::max(absmax, (float)fabs(ptr[k]));
            }

            float A_int8_scale = absmax == 0.f ? 1.f : 127.f / absmax;
            A_int8_scales[i] = A_int8_scale;

            signed char* ptrAi = A_int8.row<signed char>(i);

            for (int k = 0; k < A_int8.w; k++)
            {
                ptrAi[k] = float2int8(ptr[k] * A_int8_scale);
            }
        }
    }

    // dynamic quantize B
    Mat B0_int8 = B0;
    float B_int8_scale = B_data_int8_scale;
    if (B0_int8.elemsize != 1)
    {
        B0_int8.create(B0.w, B0.dims == 3 ? B0.c : B0.h, (size_t)1u, 1, opt.workspace_allocator);
        if (B0_int8.empty())
            return -100;

        float absmax = 0.f;
        for (int i = 0; i < B0_int8.h; i++)
        {
            const size_t B_hstep = B0.dims == 3 ? B0.cstep : (size_t)B0.w;
            const float* ptr = (const float*)B0 + i * B_hstep;

            for (int k = 0; k < B0_int8.w; k++)
            {
                absmax = std::max(absmax, (float)fabs(ptr[k]));
            }
        }

        B_int8_scale = absmax == 0.f ? 1.f : 127.f / absmax;

        for (int i = 0; i < B0_int8.h; i++)
        {
            const size_t B_hstep = B0.dims == 3 ? B0.cstep : (size_t)B0.w;
            const float* ptr = (const float*)B0 + i * B_hstep;

            signed char* ptrBi = B0_int8.row<signed char>(i);

            for (int k = 0; k < B0_int8.w; k++)
            {
                ptrBi[k] = float2int8(ptr[k] * B_int8_scale);
            }
        }
    }

    Mat BT_int8;
    if (transB == 0)
    {
        // transpose B to col-major
        BT_int8.create(B0_int8.h, B0_int8.w, (size_t)1u, 1, opt.workspace_allocator);
        if (BT_int8.empty())
            return -100;

        for (int i = 0; i < BT_int8.h; i++)
        {
            signed char* ptr = BT_int8.row<signed char>(i);
            for (int j = 0; j < BT_int8.w; j++)
            {
                ptr[j] = B0_int8.row<const signed char>(j)[i];
            }
        }
    }
    else
    {
        BT_int8 = B0_int8;
    }

    const int M = A_int8.h;
    const int N = BT_int8.h;

    Mat C;
    int broadcast_type_C = 0;
    if (constantC)
    {
        C = C_data;
        broadcast_type_C = constant_broadcast_type_C;
    }
    else
    {
        if (constantA && constantB && bottom_blobs.size() == 1)
        {
            C = bottom_blobs[0];
        }
        else if ((constantA || constantB) && bottom_blobs.size() == 2)
        {
            C = bottom_blobs[1];
        }
        else if (bottom_blobs.size() == 3)
        {
            C = bottom_blobs[2];
        }

        if (!C.empty())
        {
            if (C.dims == 1 && C.w == 1)
            {
                // scalar
                broadcast_type_C = 0;
            }
            if (C.dims == 1 && C.w == M)
            {
                // M
                // auto broadcast from h to w is the ncnn-style convention
                broadcast_type_C = 1;
            }
            if (C.dims == 1 && C.w == N)
            {
                // N
                broadcast_type_C = 4;
            }
            if (C.dims == 2 && C.w == 1 && C.h == M)
            {
                // Mx1
                broadcast_type_C = 2;
            }
            if (C.dims == 2 && C.w == N && C.h == M)
            {
                // MxN
                broadcast_type_C = 3;
            }
            if (C.dims == 2 && C.w == N && C.h == 1)
            {
                // 1xN
                broadcast_type_C = 4;
            }
        }
    }

    Mat& top_blob = top_blobs[0];
    if (output_transpose)
    {
        if (output_N1M)
            top_blob.create(M, 1, N, 4u, opt.blob_allocator);
        else
            top_blob.create(M, N, 4u, opt.blob_allocator);
    }
    else
    {
        if (output_N1M)
            top_blob.create(N, 1, M, 4u, opt.blob_allocator);
        else
            top_blob.create(N, M, 4u, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    gemm_transB_int8(A_int8, BT_int8, A_int8_scales, B_int8_scale, C, top_blob, alpha, beta, broadcast_type_C, output_transpose, opt);

    return 0;
}
#endif // NCNN_INT8

} // namespace ncnn
