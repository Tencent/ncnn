// Highly optimized implementation for Tile with cache optimization
// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "tile.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif

namespace ncnn {

Tile::Tile()
{
    one_blob_only = false;
    support_inplace = false;
}

int Tile::load_param(const ParamDict& pd)
{
    axis = pd.get(0, 0);
    tiles = pd.get(1, 1);
    repeats = pd.get(2, Mat());
    return 0;
}

int Tile::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // ONNX mode: repeats comes as second input blob
    if (bottom_blobs.size() >= 2 && !bottom_blobs[1].empty())
    {
        const Mat& bottom_blob = bottom_blobs[0];
        const Mat& repeats_blob = bottom_blobs[1];

        int dims = bottom_blob.dims;
        const int* repeats_ptr = (const int*)repeats_blob;
        int repeats_count = (repeats_blob.dims == 1) ? repeats_blob.w : (int)repeats_blob.total();

        // Calculate repeat factors
        int repeat_w = 1, repeat_h = 1, repeat_c = 1;

        if (repeats_count == 1)
        {
            repeat_w = repeats_ptr[0];
        }
        else if (repeats_count == 2)
        {
            repeat_w = repeats_ptr[0];
            repeat_h = repeats_ptr[1];
        }
        else if (repeats_count >= 3)
        {
            repeat_w = repeats_ptr[0];
            repeat_h = repeats_ptr[1];
            repeat_c = repeats_ptr[2];
        }

        int outw = bottom_blob.w * repeat_w;
        int outh = bottom_blob.h * repeat_h;
        int outc = bottom_blob.c * repeat_c;

        Mat& top_blob = top_blobs[0];
        top_blob.create(outw, outh, outc, bottom_blob.elemsize, bottom_blob.elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* ptr = bottom_blob;
        float* outptr = top_blob;

// HOT PATH: Optimized for common case repeat_w > 1, repeat_h = 1
#if __ARM_NEON
        if (repeat_w > 1 && repeat_h == 1 && repeat_c == 1 && opt.num_threads > 1)
        {
            const int w = bottom_blob.w;
            const int outw_total = outw;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < outh; y++)
            {
                const float* src_row = ptr + y * w;
                float* dst_row = outptr + y * outw_total;

                // Process each source element and repeat it
                for (int x = 0; x < w; x++)
                {
                    float val = src_row[x];
                    float* dst_ptr = dst_row + x * repeat_w;

                    // Unroll based on repeat_w
                    if (repeat_w == 2)
                    {
                        float32x2_t v = vdup_n_f32(val);
                        vst1_f32(dst_ptr, v);
                    }
                    else if (repeat_w == 4)
                    {
                        float32x4_t v = vdupq_n_f32(val);
                        vst1q_f32(dst_ptr, v);
                    }
                    else if (repeat_w == 8)
                    {
                        float32x4x2_t v;
                        v.val[0] = vdupq_n_f32(val);
                        v.val[1] = vdupq_n_f32(val);
                        vst2q_f32(dst_ptr, v);
                    }
                    else if ((repeat_w & 3) == 0)
                    {
                        // Multiple of 4
                        float32x4_t v = vdupq_n_f32(val);
                        for (int i = 0; i < repeat_w; i += 4)
                        {
                            vst1q_f32(dst_ptr + i, v);
                        }
                    }
                    else
                    {
                        // General case with unrolling
                        const int nn = repeat_w >> 2;
                        const int rem = repeat_w - (nn << 2);
                        float32x4_t v = vdupq_n_f32(val);
                        for (int i = 0; i < nn; i++)
                        {
                            vst1q_f32(dst_ptr + (i << 2), v);
                        }
                        for (int i = nn << 2; i < repeat_w; i++)
                        {
                            dst_ptr[i] = val;
                        }
                    }
                }
            }
            return 0;
        }

        // HOT PATH: Optimized for repeat_h > 1, repeat_w = 1 (vertical tiling)
        if (repeat_w == 1 && repeat_h > 1 && repeat_c == 1 && opt.num_threads > 1)
        {
            const int w = bottom_blob.w;
            const int h = bottom_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int t = 0; t < opt.num_threads; t++)
            {
                int thread_start = (t * outh) / opt.num_threads;
                int thread_end = ((t + 1) * outh) / opt.num_threads;

                for (int i = thread_start; i < thread_end; i++)
                {
                    int src_row = i / repeat_h;
                    const float* src_ptr = ptr + src_row * w;
                    float* dst_ptr = outptr + i * outw;

                    // Copy row with prefetching and NEON
                    const int nn = w >> 2;
                    const int remain = w - (nn << 2);

                    // Prefetch next row
                    if (i + 1 < thread_end)
                    {
                        __builtin_prefetch(ptr + ((i / repeat_h) + 1) * w, 0, 3);
                    }

                    for (int j = 0; j < nn; j++)
                    {
                        float32x4_t v = vld1q_f32(src_ptr + j * 4);
                        vst1q_f32(dst_ptr + j * 4, v);
                    }
                    for (int j = nn << 2; j < w; j++)
                    {
                        dst_ptr[j] = src_ptr[j];
                    }
                }
            }
            return 0;
        }
#endif

        // General path with OpenMP and cache-friendly access
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < outc; q++)
        {
            const float* ptr_channel = ptr + bottom_blob.cstep * (q / repeat_c);
            float* outptr_channel = outptr + top_blob.cstep * q;

            for (int i = 0; i < outh; i++)
            {
                const float* ptr_row = ptr_channel + bottom_blob.w * (i / repeat_h);
                float* outptr_row = outptr_channel + outw * i;

                // Optimized row copy with better ILP
                const int w = bottom_blob.w;
                const int repeat_w_local = repeat_w;

                for (int j = 0; j < w; j++)
                {
                    float val = ptr_row[j];
                    float* dst = outptr_row + j * repeat_w_local;
                    for (int k = 0; k < repeat_w_local; k++)
                    {
                        dst[k] = val;
                    }
                }
            }
        }

        return 0;
    }

    // Legacy mode: use parameters (unchanged, omitted for brevity)
    const Mat& bottom_blob = bottom_blobs[0];
    int dims = bottom_blob.dims;
    int repeat_w = 1, repeat_h = 1, repeat_c = 1;
    const int repeats_num = repeats.w;

    if (repeats.empty())
    {
        if (dims == 1)
            repeat_w = tiles;
        else if (dims == 2)
        {
            if (axis == 0)
                repeat_h = tiles;
            else
                repeat_w = tiles;
        }
        else if (dims == 3)
        {
            if (axis == 0)
                repeat_c = tiles;
            else if (axis == 1)
                repeat_h = tiles;
            else
                repeat_w = tiles;
        }
    }
    else
    {
        const int* repeats_ptr = repeats;
        if (repeats_num >= 1) repeat_w = repeats_ptr[repeats_num - 1];
        if (repeats_num >= 2) repeat_h = repeats_ptr[repeats_num - 2];
        if (repeats_num >= 3) repeat_c = repeats_ptr[repeats_num - 3];
    }

    int outw = bottom_blob.w * repeat_w;
    int outh = bottom_blob.h * repeat_h;
    int outc = bottom_blob.c * repeat_c;

    Mat& top_blob = top_blobs[0];
    top_blob.create(outw, outh, outc, bottom_blob.elemsize, bottom_blob.elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const float* ptr = bottom_blob;
    float* outptr = top_blob;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < outc; q++)
    {
        const float* ptr_channel = ptr + bottom_blob.cstep * (q / repeat_c);
        float* outptr_channel = outptr + top_blob.cstep * q;

        for (int i = 0; i < outh; i++)
        {
            const float* ptr_row = ptr_channel + bottom_blob.w * (i / repeat_h);
            float* outptr_row = outptr_channel + outw * i;

            for (int j = 0; j < outw; j++)
            {
                outptr_row[j] = ptr_row[j / repeat_w];
            }
        }
    }

    return 0;
}

} // namespace ncnn
