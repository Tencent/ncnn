// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

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
    constant_TILE_M = pd.get(20, 0);
    constant_TILE_N = pd.get(21, 0);
    constant_TILE_K = pd.get(22, 0);

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

    return 0;
}

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

        const int A0_hstep = A0.dims == 3 ? (int)A0.cstep : A0.w;

        for (int i = 0; i < A.h; i++)
        {
            float* ptr = A.row(i);
            for (int j = 0; j < A.w; j++)
            {
                ptr[j] = A0[j * A0_hstep + i];
            }
        }
    }

    Mat B;
    if (transB == 0)
    {
        // transpose B to col-major
        B.create((B0.dims == 3 ? B0.c : B0.h), B0.w, elemsize, opt.workspace_allocator);

        const int B0_hstep = B0.dims == 3 ? (int)B0.cstep : B0.w;

        for (int i = 0; i < B.h; i++)
        {
            float* ptr = B.row(i);
            for (int j = 0; j < B.w; j++)
            {
                ptr[j] = B0[j * B0_hstep + i];
            }
        }
    }
    else
    {
        B = B0;
    }

    const int M = A.dims == 3 ? A.c : A.h;
    const int K = A.w; // assert A.w == B.w
    const int N = B.dims == 3 ? B.c : B.h;

    const float* ptrC = 0;
    int broadcast_type_C = 0;
    if (constantC)
    {
        ptrC = C_data;
        broadcast_type_C = constant_broadcast_type_C;
    }
    else
    {
        if (constantA && constantB)
        {
            ptrC = bottom_blobs.size() == 1 ? bottom_blobs[0] : 0;
        }
        else if (constantA)
        {
            ptrC = bottom_blobs.size() == 2 ? bottom_blobs[1] : 0;
        }
        else if (constantB)
        {
            ptrC = bottom_blobs.size() == 2 ? bottom_blobs[1] : 0;
        }
        else
        {
            ptrC = bottom_blobs.size() == 3 ? bottom_blobs[2] : 0;
        }

        if (ptrC)
        {
            const Mat& C = bottom_blobs[bottom_blobs.size() - 1];

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

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < M; i++)
    {
        const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

        const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;
        const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

        const float* ptrA = (const float*)A + i * A_hstep;

        for (int j = 0; j < N; j++)
        {
            const float* ptrB = (const float*)B + j * B_hstep;

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
                sum += ptrA[k] * ptrB[k];
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

    return 0;
}

} // namespace ncnn
