// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "mod.h"
#include <cmath>

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

    const float* a = a_blob;
    const float* b = b_blob;
    float* out = top_blob;

    const int total = (int)top_blob.total();

    if (fmod == 0)
    {
        // Python-style modulo (remainder with same sign as divisor)
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < total; i++)
        {
            float val_a = a[i];
            float val_b = b[i];

            if (val_b == 0.0f)
            {
                out[i] = 0.0f;
            }
            else
            {
                // Python-style: result has same sign as divisor (b)
                float result = std::fmod(val_a, val_b);
                if ((result != 0.0f) && ((val_b < 0.0f) != (result < 0.0f)))
                {
                    result += val_b;
                }
                out[i] = result;
            }
        }
    }
    else
    {
        // C-style fmod (remainder with same sign as dividend)
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < total; i++)
        {
            float val_a = a[i];
            float val_b = b[i];

            if (val_b == 0.0f)
            {
                out[i] = 0.0f;
            }
            else
            {
                out[i] = std::fmod(val_a, val_b);
            }
        }
    }

    return 0;
}

} // namespace ncnn
