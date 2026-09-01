// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "datareader.h"
#include "net.h"

#include <stdio.h>
#include <string.h>
#include <vector>

// one network input blob is consumed directly by two gemm layers
//   Input in0 -> Gemm g4 (B = MemoryData B4) -> out4
//   Input in0 -> Gemm g8 (B = MemoryData B8) -> out8
// extracting both outputs with a single extractor must give valid results
static const char* param_data =
    "7767517\n"
    "5 5\n"
    "Input            in0            0 1 in0\n"
    "MemoryData       B4             0 1 B4 0=4 1=2\n"
    "MemoryData       B8             0 1 B8 0=4 1=8\n"
    "Gemm             g4             2 1 in0 B4 out4 0=1.0 1=0.0 3=1\n"
    "Gemm             g8             2 1 in0 B8 out8 0=1.0 1=0.0 3=1\n";

static ncnn::Mat gemm_ref(const ncnn::Mat& A, const ncnn::Mat& B)
{
    // out = A * B^T with A(w=K,h=M) B(w=K,h=N) -> out(w=N,h=M)
    const int M = A.h;
    const int N = B.h;
    const int K = A.w;

    ncnn::Mat out(N, M);
    for (int i = 0; i < M; i++)
    {
        const float* aptr = A.row(i);
        float* optr = out.row(i);
        for (int j = 0; j < N; j++)
        {
            const float* bptr = B.row(j);
            float sum = 0.f;
            for (int k = 0; k < K; k++)
            {
                sum += aptr[k] * bptr[k];
            }
            optr[j] = sum;
        }
    }

    return out;
}

static int test_extractor_shared_input_multi_extract()
{
    const int M = 3;
    const int K = 4;
    const int N4 = 2;
    const int N8 = 8;

    ncnn::Mat A = RandomMat(K, M);
    ncnn::Mat B4 = RandomMat(K, N4);
    ncnn::Mat B8 = RandomMat(K, N8);

    // memorydata weights are read in layer order: B4 then B8
    std::vector<unsigned char> model_bin(B4.total() * B4.elemsize + B8.total() * B8.elemsize);
    memcpy(model_bin.data(), B4.data, B4.total() * B4.elemsize);
    memcpy(model_bin.data() + B4.total() * B4.elemsize, B8.data, B8.total() * B8.elemsize);

    ncnn::Net net;

    int ret = net.load_param_mem(param_data);
    if (ret != 0)
    {
        fprintf(stderr, "load_param_mem failed %d\n", ret);
        return -1;
    }

    const unsigned char* mem = model_bin.data();
    ncnn::DataReaderFromMemory dr(mem);
    ret = net.load_model(dr);
    if (ret != 0)
    {
        fprintf(stderr, "load_model failed %d\n", ret);
        return -1;
    }

    ncnn::Extractor ex = net.create_extractor();

    ret = ex.input("in0", A);
    if (ret != 0)
    {
        fprintf(stderr, "input failed %d\n", ret);
        return -1;
    }

    ncnn::Mat out4;
    ret = ex.extract("out4", out4);
    if (ret != 0)
    {
        fprintf(stderr, "extract out4 failed %d\n", ret);
        return -1;
    }

    ncnn::Mat out8;
    ret = ex.extract("out8", out8);
    if (ret != 0)
    {
        fprintf(stderr, "extract out8 failed %d\n", ret);
        return -1;
    }

    if (out4.dims != 2 || out4.w != N4 || out4.h != M)
    {
        fprintf(stderr, "out4 shape mismatch: dims=%d w=%d h=%d, expected dims=2 w=%d h=%d\n", out4.dims, out4.w, out4.h, N4, M);
        return -1;
    }

    if (out8.dims != 2 || out8.w != N8 || out8.h != M)
    {
        fprintf(stderr, "out8 shape mismatch: dims=%d w=%d h=%d, expected dims=2 w=%d h=%d\n", out8.dims, out8.w, out8.h, N8, M);
        return -1;
    }

    ncnn::Mat ref4 = gemm_ref(A, B4);
    ncnn::Mat ref8 = gemm_ref(A, B8);

    ret = CompareMat(out4, ref4, 0.001);
    if (ret != 0)
    {
        fprintf(stderr, "out4 value mismatch\n");
        return ret;
    }

    ret = CompareMat(out8, ref8, 0.001);
    if (ret != 0)
    {
        fprintf(stderr, "out8 value mismatch\n");
        return ret;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return test_extractor_shared_input_multi_extract();
}
