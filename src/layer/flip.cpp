// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "flip.h"

namespace ncnn {

Flip::Flip()
{
    one_blob_only = true;
}

int Flip::load_param(const ParamDict& pd)
{
    axes = pd.get(0, Mat());

    if (axes.w > 4)
    {
        // only handle up to 4-dim
        return -1;
    }

    return 0;
}

int Flip::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (axes.empty())
    {
        top_blob = bottom_blob;
        return 0;
    }

    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;

    int axes_flag[4] = {0};
    bool flip_w = false;
    bool flip_h = false;
    bool flip_d = false;
    bool flip_c = false;
    {
        const int* axes_ptr = axes;
        for (int i = 0; i < axes.w; i++)
        {
            int axis = axes_ptr[i];
            // handle negative axis
            if (axis < 0)
                axis += dims;
            axes_flag[axis] = 1;
        }

        if (dims == 1)
        {
            flip_w = true;
        }
        else if (dims == 2)
        {
            if (axes_flag[0] == 1) flip_h = true;
            if (axes_flag[1] == 1) flip_w = true;
        }
        else if (dims == 3)
        {
            if (axes_flag[0] == 1) flip_c = true;
            if (axes_flag[1] == 1) flip_h = true;
            if (axes_flag[2] == 1) flip_w = true;
        }
        else if (dims == 4)
        {
            if (axes_flag[0] == 1) flip_c = true;
            if (axes_flag[1] == 1) flip_d = true;
            if (axes_flag[2] == 1) flip_h = true;
            if (axes_flag[3] == 1) flip_w = true;
        }
    }

    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        for (int z = 0; z < d; z++)
        {
            for (int i = 0; i < h; i++)
            {
                int q2 = flip_c ? channels - 1 - q : q;
                int z2 = flip_d ? d - 1 - z : z;
                int i2 = flip_h ? h - 1 - i : i;

                const float* ptr = bottom_blob.channel(q2).depth(z2).row(i2);
                float* outptr = top_blob.channel(q).depth(z).row(i);

                if (flip_w)
                {
                    ptr += w - 1;
                    for (int j = 0; j < w; j++)
                    {
                        *outptr++ = *ptr--;
                    }
                }
                else
                {
                    memcpy(outptr, ptr, w * sizeof(float));
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
