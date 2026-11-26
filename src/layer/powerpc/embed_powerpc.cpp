// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "embed_powerpc.h"

#include <string.h>

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Embed_powerpc::Embed_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

#if __ALTIVEC__
// vectorized fp32 embedding lookup kernel, identical semantics to the generic
// implementation with a 4-wide elementwise inner loop for the bias add / copy
static void embed_fp32(const Mat& bottom_blob, const Mat& weight_data, const Mat& bias_data, Mat& top_blob, int input_dim, const Option& opt)
{
    const int num_output = top_blob.w;
    const int words = top_blob.h;

    const float* bias_ptr = bias_data;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < words; q++)
    {
        float* outptr = top_blob.row(q);

        int word_index = ((const int*)bottom_blob)[q];

        if (word_index < 0)
            word_index = 0;
        if (word_index >= input_dim)
            word_index = input_dim - 1;

        const float* em = (const float*)weight_data + num_output * word_index;

        if (bias_ptr)
        {
            int p = 0;
            for (; p + 4 <= num_output; p += 4)
            {
                __ncnn_altivec_store(outptr + p, vec_add(__ncnn_altivec_load(em + p), __ncnn_altivec_load(bias_ptr + p)));
            }
            for (; p < num_output; p++)
            {
                outptr[p] = em[p] + bias_ptr[p];
            }
        }
        else
        {
            int p = 0;
            for (; p + 4 <= num_output; p += 4)
            {
                __ncnn_altivec_store(outptr + p, __ncnn_altivec_load(em + p));
            }
            for (; p < num_output; p++)
            {
                outptr[p] = em[p];
            }
        }
    }
}
#endif // __ALTIVEC__

int Embed_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if __ALTIVEC__
    // the embedding table is fp32, vectorize the fp32 elementwise inner loops
    // note: the bottom blob is a 1D int32 word index blob and is never packed
    if (weight_data.elemsize == 4u && bottom_blob.elempack == 1)
    {
        const int words = bottom_blob.w;

        top_blob.create(num_output, words, 4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        embed_fp32(bottom_blob, weight_data, bias_data, top_blob, input_dim, opt);

        return 0;
    }
#endif // __ALTIVEC__

    // int8 or non-altivec, delegate to the generic implementation
    if (bottom_blob.elempack != 1 && bottom_blob.elembits() == 32)
    {
        Mat bottom_blob_unpacked;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
        if (bottom_blob_unpacked.empty())
            return -100;

        return Embed::forward(bottom_blob_unpacked, top_blob, opt);
    }

    return Embed::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
