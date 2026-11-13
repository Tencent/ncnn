// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "swiglu.h"

#include <float.h>

namespace ncnn {

SwiGLU::SwiGLU()
{
    one_blob_only = false;
    support_inplace = false;
}

int SwiGLU::load_param(const ParamDict& pd)
{
    sigmoid_scale = pd.get(0, 1.f);
    up_offset = pd.get(1, 0.f);
    activation_limit = pd.get(2, 0.f);

    return 0;
}

int SwiGLU::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.size() != 2)
        return -100;

    const Mat& gate = bottom_blobs[0];
    const Mat& up = bottom_blobs[1];

    if (gate.dims != up.dims || gate.w != up.w || gate.h != up.h || gate.d != up.d || gate.c != up.c || gate.elempack != up.elempack || gate.elembits() != up.elembits())
        return -100;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(gate, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int channels = gate.c;
    const int size = gate.w * gate.h * gate.d * gate.elempack;
    const float climit = activation_limit > 0.f ? activation_limit : FLT_MAX;
    const float mlimit = -climit;

    // gather short channels into one task and split long channels, so that
    // 2d llm tensors with c == 1 also spread over all threads
    const int block_size = 1024;
    const int nn_block = (size + block_size - 1) / block_size;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ii = 0; ii < channels * nn_block; ii++)
    {
        const int q = ii / nn_block;
        const int offset = ii % nn_block * block_size;
        const int len = std::min(size - offset, block_size);

        const float* gateptr = (const float*)gate.channel(q) + offset;
        const float* upptr = (const float*)up.channel(q) + offset;
        float* outptr = (float*)top_blob.channel(q) + offset;

        for (int i = 0; i < len; i++)
        {
            float g = gateptr[i];
            float u = upptr[i];

            g = std::min(g, climit);
            u = std::min(std::max(u, mlimit), climit);

            outptr[i] = g / (1.f + expf(-sigmoid_scale * g)) * (u + up_offset);
        }
    }

    return 0;
}

} // namespace ncnn
