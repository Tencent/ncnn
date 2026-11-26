// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "../tools/quantize/ncnndit_quant.h"

#include <memory>

static bool same_data(const ncnn::Mat& a, const ncnn::Mat& b)
{
    return !a.empty() && !b.empty() && a.w == b.w && a.h == b.h && a.elemsize == b.elemsize
           && memcmp(a.data, b.data, (size_t)a.w * a.h * a.elemsize) == 0;
}

static int test_dit_quantize_square(int bits)
{
    const int size = 4;
    ncnn::Mat weights[2] = {ncnn::Mat(size, size), ncnn::Mat(size, size)};
    for (int n = 0; n < size; n++)
    {
        for (int k = 0; k < size; k++)
        {
            const float v = (float)(9 * n - 2 * k + 1);
            weights[0].row(k)[n] = v;
            weights[1].row(n)[k] = v;
        }
    }

    DiTQuantizeParams params;
    params.bits = bits;
    params.block_size = 32;

    ncnn::Mat scales[2];
    ncnn::Mat codes[2];
    for (int transB = 0; transB < 2; transB++)
    {
        std::unique_ptr<ncnn::Layer> layer(ncnn::create_layer_naive("Gemm"));
        ncnn::Gemm* gemm = static_cast<ncnn::Gemm*>(layer.get());
        gemm->transB = transB;
        gemm->constantK = size;
        gemm->constantN = size;
        gemm->B_data = weights[transB].clone();

        ncnn::Mat table_scales;
        ncnn::Mat table_codes;
        if (quantize_dit_weight(gemm, params, table_scales, 0, 2) != 0
                || quantize_dit_weight(gemm, params, table_scales, &table_codes, 2) != 0
                || quantize_dit_weight(gemm, params, scales[transB], &codes[transB], 2) != 0
                || !same_data(gemm->B_data, weights[transB])
                || !same_data(table_scales, scales[transB]) || !same_data(table_codes, codes[transB]))
        {
            fprintf(stderr, "test_dit_quantize_square failed bits=%d transB=%d\n", bits, transB);
            return -1;
        }
    }

    if (!same_data(scales[0], scales[1]) || !same_data(codes[0], codes[1]))
    {
        fprintf(stderr, "test_dit_quantize_square layout mismatch bits=%d\n", bits);
        return -1;
    }

    return 0;
}

int main()
{
    return test_dit_quantize_square(4) || test_dit_quantize_square(6);
}
