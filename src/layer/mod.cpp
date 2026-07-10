// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "mod.h"

#include <math.h>

namespace ncnn {

Mod::Mod()
{
    one_blob_only = false;
    support_inplace = false;
    fmod = 0;
}

int Mod::load_param(const ParamDict& pd)
{
    fmod = pd.get(0, 0);

    return 0;
}

int Mod::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.size() < 2)
        return -1;

    const Mat& a_blob = bottom_blobs[0];
    const Mat& b_blob = bottom_blobs[1];

    // Output has same shape as a_blob
    const Mat& out_shape = a_blob;

    Mat& top_blob = top_blobs[0];
    top_blob.create(out_shape.w, out_shape.h, out_shape.c, a_blob.elemsize, a_blob.elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int out_w = top_blob.w;
    const int out_h = top_blob.h;
    const int out_c = top_blob.c;

    const int count = out_h * out_w; // contiguous elements per channel slice

    if (fmod == 0)
    {
        // Python-style modulo (remainder with same sign as divisor)
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int z = 0; z < out_c; z++)
        {
            const float* aptr = (const float*)a_blob + z * (int)a_blob.cstep;
            const float* bptr = (const float*)b_blob + z * (int)b_blob.cstep;
            float* optr = (float*)top_blob + z * (int)top_blob.cstep;
            for (int i = 0; i < count; i++)
            {
                const float val_b = bptr[i];
                if (val_b == 0.0f)
                {
                    optr[i] = 0.0f;
                }
                else
                {
                    float result = ::fmodf(aptr[i], val_b);
                    if ((result != 0.0f) && ((val_b < 0.0f) != (result < 0.0f)))
                        result += val_b;
                    optr[i] = result;
                }
            }
        }
    }
    else
    {
        // C-style fmod (remainder with same sign as dividend)
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int z = 0; z < out_c; z++)
        {
            const float* aptr = (const float*)a_blob + z * (int)a_blob.cstep;
            const float* bptr = (const float*)b_blob + z * (int)b_blob.cstep;
            float* optr = (float*)top_blob + z * (int)top_blob.cstep;
            for (int i = 0; i < count; i++)
            {
                const float val_b = bptr[i];
                optr[i] = (val_b == 0.0f) ? 0.0f : ::fmodf(aptr[i], val_b);
            }
        }
    }

    return 0;
}

} // namespace ncnn
