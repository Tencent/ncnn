// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "tile_powerpc.h"

#include <string.h>

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Tile_powerpc::Tile_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

int Tile_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elempack = bottom_blob.elempack;

    if (elempack != 1)
    {
        // output dims may change (outdims can exceed input dims), so the packing axis
        // of the input and output can differ; operate on the unpacked layout and
        // re-pack the result afterwards
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

        // re-pack if the packing axis of the output is exactly divisible
        int repack_count = 0;
        if (top_blob_unpacked.dims == 1) repack_count = top_blob_unpacked.w;
        if (top_blob_unpacked.dims == 2) repack_count = top_blob_unpacked.h;
        if (top_blob_unpacked.dims == 3 || top_blob_unpacked.dims == 4) repack_count = top_blob_unpacked.c;

        if (repack_count % 4 == 0)
        {
            Mat top_blob_packed;
            convert_packing(top_blob_unpacked, top_blob_packed, 4, opt);
            // the count is divisible by 4, so an empty result is an allocation
            // failure. falling through to the unpacked blob would hide the oom.
            if (top_blob_packed.empty())
                return -100;

            top_blob = top_blob_packed;
            return 0;
        }

        top_blob = top_blob_unpacked;
        return 0;
    }

    int dims = bottom_blob.dims;
    int repeat_w = 1;
    int repeat_h = 1;
    int repeat_d = 1;
    int repeat_c = 1;

    const int repeats_num = repeats.w;

    if (repeats.empty())
    {
        if (dims == 1) // axis == 0
        {
            repeat_w = tiles;
        }
        else if (dims == 2)
        {
            if (axis == 0) repeat_h = tiles;
            if (axis == 1) repeat_w = tiles;
        }
        else if (dims == 3)
        {
            if (axis == 0) repeat_c = tiles;
            if (axis == 1) repeat_h = tiles;
            if (axis == 2) repeat_w = tiles;
        }
        else if (dims == 4)
        {
            if (axis == 0) repeat_c = tiles;
            if (axis == 1) repeat_d = tiles;
            if (axis == 2) repeat_h = tiles;
            if (axis == 3) repeat_w = tiles;
        }
    }
    else
    {
        // numpy style tile
        const int* repeats_ptr = repeats;

        if (repeats_num == 1)
        {
            repeat_w = repeats_ptr[0];
        }
        if (repeats_num == 2)
        {
            repeat_h = repeats_ptr[0];
            repeat_w = repeats_ptr[1];
        }
        if (repeats_num == 3)
        {
            if (dims == 4)
            {
                repeat_d = repeats_ptr[0];
                repeat_h = repeats_ptr[1];
                repeat_w = repeats_ptr[2];
            }
            else
            {
                repeat_c = repeats_ptr[0];
                repeat_h = repeats_ptr[1];
                repeat_w = repeats_ptr[2];
            }
        }
        if (repeats_num == 4)
        {
            repeat_c = repeats_ptr[0];
            repeat_d = repeats_ptr[1];
            repeat_h = repeats_ptr[2];
            repeat_w = repeats_ptr[3];
        }
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    const int outdims = std::max(dims, repeats_num);

    if (repeat_w == 1 && repeat_h == 1 && repeat_d == 1 && repeat_c == 1)
    {
        // all ones
        if (repeats_num == 0 || dims == repeats_num)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }

    int outw = w * repeat_w;
    int outh = h * repeat_h;
    int outd = d * repeat_d;
    int outc = channels * repeat_c;
    if (outdims == 1)
    {
        top_blob.create(outw, elemsize, elempack, opt.blob_allocator);
    }
    if (outdims == 2)
    {
        top_blob.create(outw, outh, elemsize, elempack, opt.blob_allocator);
    }
    if (outdims == 3)
    {
        top_blob.create(outw, outh, outc, elemsize, elempack, opt.blob_allocator);
    }
    if (outdims == 4)
    {
        top_blob.create(outw, outh, outd, outc, elemsize, elempack, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    const int row_size = w * elempack; // contiguous elements per row

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        // repeat 0-w
        for (int z = 0; z < d; z++)
        {
            for (int y = 0; y < h; y++)
            {
                const float* ptr = bottom_blob.channel(q).depth(z).row(y);
                float* outptr = top_blob.channel(q).depth(z).row(y);

                for (int p = 0; p < repeat_w; p++)
                {
                    int i = 0;
#if __ALTIVEC__
                    for (; i + 3 < row_size; i += 4)
                    {
                        __ncnn_altivec_store(outptr, __ncnn_altivec_load(ptr));
                        ptr += 4;
                        outptr += 4;
                    }
#endif // __ALTIVEC__
                    for (; i < row_size; i++)
                    {
                        *outptr++ = *ptr++;
                    }

                    ptr -= row_size;
                }
            }
        }

        // repeat 1-h
        for (int z = 0; z < d; z++)
        {
            const float* ptr = top_blob.channel(q).depth(z);
            float* outptr = top_blob.channel(q).depth(z).row(h);

            const int size = w * repeat_w * h * elempack;
            for (int p = 1; p < repeat_h; p++)
            {
                memcpy(outptr, ptr, size * sizeof(float));
                outptr += size;
            }
        }

        // repeat 1-d
        {
            const float* ptr = top_blob.channel(q);
            float* outptr = top_blob.channel(q).depth(d);

            const int size = w * repeat_w * h * repeat_h * d * elempack;
            for (int p = 1; p < repeat_d; p++)
            {
                memcpy(outptr, ptr, size * sizeof(float));
                outptr += size;
            }
        }
    }

    // repeat 1-c
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 1; p < repeat_c; p++)
    {
        const float* ptr = top_blob.channel_range(0, channels);
        float* outptr = top_blob.channel_range(p * channels, channels);

        memcpy(outptr, ptr, top_blob.cstep * channels * elemsize);
    }

    return 0;
}

} // namespace ncnn
