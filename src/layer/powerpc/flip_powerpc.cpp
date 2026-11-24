// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "flip_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Flip_powerpc::Flip_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

#if __ALTIVEC__
// vectorized contiguous copy of n floats
static void copy_vec(float* dst, const float* src, int n)
{
    int i = 0;
    for (; i + 4 <= n; i += 4)
    {
        v4f32 _v = __ncnn_altivec_load(src + i);
        __ncnn_altivec_store(dst + i, _v);
    }
    for (; i < n; i++)
    {
        dst[i] = src[i];
    }
}
#endif // __ALTIVEC__

int Flip_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elempack != 1)
    {
        // flip keeps the shape, so the packing axis is preserved; operate on the
        // unpacked layout and re-pack afterwards
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        Mat bottom_blob_unpacked;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
        if (bottom_blob_unpacked.empty())
            return -100;

        Mat top_blob_unpacked;
        int ret = forward(bottom_blob_unpacked, top_blob_unpacked, opt);
        if (ret != 0)
            return ret;

        int repack_count = 0;
        if (top_blob_unpacked.dims == 1)
            repack_count = top_blob_unpacked.w;
        if (top_blob_unpacked.dims == 2)
            repack_count = top_blob_unpacked.h;
        if (top_blob_unpacked.dims == 3 || top_blob_unpacked.dims == 4)
            repack_count = top_blob_unpacked.c;

        if (opt.use_packing_layout && repack_count % 4 == 0)
        {
            Mat top_blob_packed;
            convert_packing(top_blob_unpacked, top_blob_packed, 4, opt);
            if (!top_blob_packed.empty())
            {
                top_blob = top_blob_packed;
                return 0;
            }
        }

        top_blob = top_blob_unpacked;
        return 0;
    }

    if (bottom_blob.elembits() != 32)
    {
        // int8 / bf16 -> base implementation
        return Flip::forward(bottom_blob, top_blob, opt);
    }

#if __ALTIVEC__
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
                    // reversed copy along w, scalar
                    ptr += w - 1;
                    for (int j = 0; j < w; j++)
                    {
                        *outptr++ = *ptr--;
                    }
                }
                else
                {
                    // contiguous row copy, vectorize
                    copy_vec(outptr, ptr, w);
                }
            }
        }
    }

    return 0;
#else
    // no altivec -> base implementation
    return Flip::forward(bottom_blob, top_blob, opt);
#endif // __ALTIVEC__
}

} // namespace ncnn
