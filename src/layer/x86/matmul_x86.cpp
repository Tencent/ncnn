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

#include "matmul_x86.h"

#include "layer_type.h"

namespace ncnn {

MatMul_x86::MatMul_x86()
{
    gemm = 0;
}

int MatMul_x86::create_pipeline(const Option& opt)
{
    gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);

    ncnn::ParamDict pd;
    pd.set(2, 0);      // transA
    pd.set(3, transB); // transB
    pd.set(4, 0);      // constantA
    pd.set(5, 0);      // constantB
    pd.set(6, 1);      // constantC
    pd.set(7, 0);      // M = outch
    pd.set(8, 0);      // N = size
    pd.set(9, 0);      // K = maxk*inch
    pd.set(10, -1);    // constant_broadcast_type_C = null
    pd.set(11, 0);     // output_N1M
    pd.set(12, 1);     // output_elempack

    gemm->load_param(pd);

    gemm->load_model(ModelBinFromMatArray(0));

    gemm->create_pipeline(opt);

    return 0;
}

int MatMul_x86::destroy_pipeline(const Option& opt)
{
    if (gemm)
    {
        gemm->destroy_pipeline(opt);
        delete gemm;
        gemm = 0;
    }

    return 0;
}

int MatMul_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
        std::vector<Mat> _bottom_blobs(2);
        _bottom_blobs[0] = A.reshape(A.w, 1);
        _bottom_blobs[1] = transB ? B.reshape(B.w, 1) : B.reshape(1, B.w);
        gemm->forward(_bottom_blobs, top_blobs, opt);

        top_blob = top_blob.reshape(1, opt.blob_allocator);
    }
    else if (Adims == 2 && Bdims == 2)
    {
        // matrix multiply
        gemm->forward(bottom_blobs, top_blobs, opt);
    }
    else if (Adims == 1 && Bdims == 2)
    {
        // matrix multiply
        std::vector<Mat> _bottom_blobs(2);
        _bottom_blobs[0] = A.reshape(A.w, 1);
        _bottom_blobs[1] = B;
        gemm->forward(_bottom_blobs, top_blobs, opt);

        top_blob = top_blob.reshape(top_blob.w, opt.blob_allocator);
    }
    else if (Adims == 2 && Bdims == 1)
    {
        // matrix multiply
        std::vector<Mat> _bottom_blobs(2);
        _bottom_blobs[0] = A;
        _bottom_blobs[1] = transB ? B.reshape(B.w, 1) : B.reshape(1, B.w);
        gemm->forward(_bottom_blobs, top_blobs, opt);

        top_blob = top_blob.reshape(top_blob.h, opt.blob_allocator);
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
            std::vector<Mat> _bottom_blobs(2);
            _bottom_blobs[0] = A1;
            _bottom_blobs[1] = B1.channel(p);
            std::vector<Mat> _top_blobs(1);
            _top_blobs[0] = top_blob1.channel(p);
            gemm->forward(_bottom_blobs, _top_blobs, opt);
        }

        if (Bdims == 3)
            top_blob = top_blob1.reshape(N, B.d * B.c, opt.blob_allocator);
        else
            top_blob = top_blob1.reshape(N, B.d, B.c, opt.blob_allocator);
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
        Mat BT = transB ? B.reshape(B.w, 1) : B.reshape(1, B.w);

        for (int p = 0; p < batch_size; p++)
        {
            std::vector<Mat> _bottom_blobs(2);
            _bottom_blobs[0] = A1.channel(p);
            _bottom_blobs[1] = BT;
            std::vector<Mat> _top_blobs(1);
            _top_blobs[0] = top_blob1.channel(p);
            gemm->forward(_bottom_blobs, _top_blobs, opt);
        }

        if (Adims == 3)
            top_blob = top_blob1.reshape(M, A.d * A.c, opt.blob_allocator);
        else
            top_blob = top_blob1.reshape(M, A.d, A.c, opt.blob_allocator);
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

        for (int p = 0; p < batch_size; p++)
        {
            int Ap = A1.c == 1 ? 0 : p;
            int Bp = B1.c == 1 ? 0 : p;

            std::vector<Mat> _bottom_blobs(2);
            _bottom_blobs[0] = A1.channel(Ap);
            _bottom_blobs[1] = B1.channel(Bp);
            std::vector<Mat> _top_blobs(1);
            _top_blobs[0] = top_blob.channel(p);
            gemm->forward(_bottom_blobs, _top_blobs, opt);
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

        for (int p = 0; p < batch_size_c; p++)
        {
            int Ap = A1.c == 1 ? 0 : p;
            int Bp = B1.c == 1 ? 0 : p;

            for (int q = 0; q < batch_size_d; q++)
            {
                int Ad = A1.d == 1 ? 0 : q;
                int Bd = B1.d == 1 ? 0 : q;

                std::vector<Mat> _bottom_blobs(2);
                _bottom_blobs[0] = A1.channel(Ap).depth(Ad);
                _bottom_blobs[1] = B1.channel(Bp).depth(Bd);
                std::vector<Mat> _top_blobs(1);
                _top_blobs[0] = top_blob.channel(p).depth(q);
                gemm->forward(_bottom_blobs, _top_blobs, opt);
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
