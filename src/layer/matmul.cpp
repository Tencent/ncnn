// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "matmul.h"

namespace ncnn {

MatMul::MatMul()
{
    one_blob_only = false;
    support_inplace = false;
}

int MatMul::load_param(const ParamDict& pd)
{
    transB = pd.get(0, 0);

    return 0;
}

static void transpose(const Mat& X, Mat& XT, const Option& opt)
{
    const int w = X.w;
    const int h = X.h;

    const float* pX = X;
    float* pXT = XT;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < w; i++)
    {
        float* ptr = pXT + i * h;
        for (int j = 0; j < h; j++)
        {
            ptr[j] = pX[j * w + i];
        }
    }
}

static void matmul_transb(const Mat& A, const Mat& B, Mat& top_blob, const Option& opt)
{
    const int M = A.h;
    const int K = A.w; // assert A.w == B.w
    const int N = B.h;

    const float* pA = A;
    const float* pB = B;
    float* pOut = top_blob;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < M; i++)
    {
        const float* ptrA = pA + i * K;
        float* outptr = pOut + i * N;

        for (int j = 0; j < N; j++)
        {
            const float* ptrB = pB + j * K;

            float sum = 0.f;
            for (int k = 0; k < K; k++)
            {
                sum += ptrA[k] * ptrB[k];
            }

            *outptr++ = sum;
        }
    }
}

int MatMul::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& A = bottom_blobs[0];
    const Mat& B = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    const int Adims = A.dims;
    const int Bdims = B.dims;
    const int max_ABdims = std::max(Adims, Bdims);
    const size_t elemsize = A.elemsize;

    if (Adims == 1 && Bdims == 1)
    {
        // dot product
        top_blob.create(1, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int K = A.w; // assert A.w == B.w
        const float* ptrA = A;
        const float* ptrB = B;

        float sum = 0.f;
        for (int k = 0; k < K; k++)
        {
            sum += ptrA[k] * ptrB[k];
        }

        top_blob[0] = sum;
    }
    else if (Adims == 2 && Bdims == 2)
    {
        // matrix multiply
        const int M = A.h;
        const int N = transB == 0 ? B.w : B.h;

        top_blob.create(N, M, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        Mat BT;
        if (transB == 0)
        {
            BT.create(B.h, B.w, elemsize, opt.workspace_allocator);
            if (BT.empty())
                return -100;

            transpose(B, BT, opt);
        }
        else
        {
            BT = B;
        }

        matmul_transb(A, BT, top_blob, opt);
    }
    else if (Adims == 1 && Bdims == 2)
    {
        // matrix multiply
        const int N = transB == 0 ? B.w : B.h;

        Mat top_blob1(N, 1, elemsize, opt.blob_allocator);
        if (top_blob1.empty())
            return -100;

        Mat A1 = A.reshape(A.w, 1);

        Mat BT;
        if (transB == 0)
        {
            BT.create(B.h, B.w, elemsize, opt.workspace_allocator);
            if (BT.empty())
                return -100;

            transpose(B, BT, opt);
        }
        else
        {
            BT = B;
        }

        matmul_transb(A1, BT, top_blob1, opt);

        top_blob = top_blob1.reshape(N);
    }
    else if (Adims == 2 && Bdims == 1)
    {
        // matrix multiply
        const int M = A.h;

        Mat top_blob1(1, M, elemsize, opt.blob_allocator);
        if (top_blob1.empty())
            return -100;

        Mat BT = B.reshape(B.w, 1);

        matmul_transb(A, BT, top_blob1, opt);

        top_blob = top_blob1.reshape(M);
    }
    else if (Adims == 1 && Bdims > 2)
    {
        // batched matrix multiply
        const int N = transB == 0 ? B.w : B.h;
        const int batch_size = B.d * B.c;

        Mat top_blob1(N, 1, batch_size, elemsize, opt.blob_allocator);
        if (top_blob1.empty())
            return -100;

        Mat A1 = A.reshape(A.w, 1);
        Mat B1 = B.reshape(B.w, B.h, batch_size);

        for (int p = 0; p < batch_size; p++)
        {
            Mat BT;
            if (transB == 0)
            {
                BT.create(B.h, B.w, elemsize, opt.workspace_allocator);
                if (BT.empty())
                    return -100;

                transpose(B1.channel(p), BT, opt);
            }
            else
            {
                BT = B1.channel(p);
            }

            Mat top_blob1_p = top_blob1.channel(p);
            matmul_transb(A1, BT, top_blob1_p, opt);
        }

        if (Bdims == 3)
            top_blob = top_blob1.reshape(N, B.d * B.c);
        else
            top_blob = top_blob1.reshape(N, B.d, B.c);
    }
    else if (Adims > 2 && Bdims == 1)
    {
        // batched matrix multiply
        const int M = A.h;
        const int batch_size = A.d * A.c;

        Mat top_blob1(1, M, batch_size, elemsize, opt.blob_allocator);
        if (top_blob1.empty())
            return -100;

        Mat A1 = A.reshape(A.w, A.h, batch_size);
        Mat BT = B.reshape(B.w, 1);

        for (int p = 0; p < batch_size; p++)
        {
            Mat top_blob1_p = top_blob1.channel(p);
            matmul_transb(A1.channel(p), BT, top_blob1_p, opt);
        }

        if (Adims == 3)
            top_blob = top_blob1.reshape(M, A.d * A.c);
        else
            top_blob = top_blob1.reshape(M, A.d, A.c);
    }
    else if (max_ABdims == 3)
    {
        Mat A1 = Adims == 2 ? A.reshape(A.w, A.h, 1) : A;
        Mat B1 = Bdims == 2 ? B.reshape(B.w, B.h, 1) : B;

        const int M = A1.h;
        const int N = transB == 0 ? B1.w : B1.h;
        const int batch_size = std::max(A1.c, B1.c);

        top_blob.create(N, M, batch_size, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        Mat BT0;
        if (B1.c == 1)
        {
            if (transB == 0)
            {
                BT0.create(B1.h, B1.w, elemsize, opt.workspace_allocator);
                if (BT0.empty())
                    return -100;

                transpose(B1.channel(0), BT0, opt);
            }
            else
            {
                BT0 = B1.channel(0);
            }
        }

        for (int p = 0; p < batch_size; p++)
        {
            int Ap = A1.c == 1 ? 0 : p;
            int Bp = B1.c == 1 ? 0 : p;

            Mat BT;
            if (B1.c == 1)
            {
                BT = BT0;
            }
            else
            {
                if (transB == 0)
                {
                    BT.create(B1.h, B1.w, elemsize, opt.workspace_allocator);
                    if (BT.empty())
                        return -100;

                    transpose(B1.channel(Bp), BT, opt);
                }
                else
                {
                    BT = B1.channel(Bp);
                }
            }

            Mat top_blob_p = top_blob.channel(p);
            matmul_transb(A1.channel(Ap), BT, top_blob_p, opt);
        }
    }
    else if (max_ABdims == 4)
    {
        Mat A1 = Adims == 3 ? A.reshape(A.w, A.h, A.c, 1) : A;
        Mat B1 = Bdims == 3 ? B.reshape(B.w, B.h, B.c, 1) : B;

        const int M = A1.h;
        const int N = transB == 0 ? B1.w : B1.h;
        const int batch_size_d = std::max(A1.d, B1.d);
        const int batch_size_c = std::max(A1.c, B1.c);

        top_blob.create(N, M, batch_size_d, batch_size_c, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        Mat BT00;
        if (B1.d == 1 && B1.c == 1)
        {
            if (transB == 0)
            {
                BT00.create(B1.h, B1.w, elemsize, opt.workspace_allocator);
                if (BT00.empty())
                    return -100;

                transpose(B1.channel(0).depth(0), BT00, opt);
            }
            else
            {
                BT00 = B1.channel(0).depth(0);
            }
        }

        for (int p = 0; p < batch_size_c; p++)
        {
            int Ap = A1.c == 1 ? 0 : p;
            int Bp = B1.c == 1 ? 0 : p;

            Mat BT0x;
            if (B1.d == 1 && B1.c != 1)
            {
                if (transB == 0)
                {
                    BT0x.create(B1.h, B1.w, elemsize, opt.workspace_allocator);
                    if (BT0x.empty())
                        return -100;

                    transpose(B1.channel(Bp).depth(0), BT0x, opt);
                }
                else
                {
                    BT0x = B1.channel(Bp).depth(0);
                }
            }

            for (int q = 0; q < batch_size_d; q++)
            {
                int Ad = A1.d == 1 ? 0 : q;
                int Bd = B1.d == 1 ? 0 : q;

                Mat BT;
                if (B1.d == 1 && B1.c == 1)
                {
                    BT = BT00;
                }
                else if (B1.d == 1 && B1.c != 1)
                {
                    BT = BT0x;
                }
                else
                {
                    if (transB == 0)
                    {
                        BT.create(B1.h, B1.w, elemsize, opt.workspace_allocator);
                        if (BT.empty())
                            return -100;

                        transpose(B1.channel(Bp).depth(Bd), BT, opt);
                    }
                    else
                    {
                        BT = B1.channel(Bp).depth(Bd);
                    }
                }

                Mat top_blob_p_q = top_blob.channel(p).depth(q);
                matmul_transb(A1.channel(Ap).depth(Ad), BT, top_blob_p_q, opt);
            }
        }
    }
    else
    {
        NCNN_LOGE("impossible matmul %d %d", Adims, Bdims);
        return -1;
    }

    return 0;
}

} // namespace ncnn
