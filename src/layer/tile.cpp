// ARM NEON optimized implementation for Tile
// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "tile.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif

namespace ncnn {

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

        // Calculate repeat factors for each dimension
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

        // ARM NEON optimized path for simple tiling
        #if __ARM_NEON
        if (repeat_w == 1 && repeat_h > 1 && repeat_c == 1 && opt.num_threads > 1)
        {
            // Optimize for vertical tiling only
            const int rows_per_thread = outh / opt.num_threads;
            
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int t = 0; t < opt.num_threads; t++)
            {
                int row_start = t * rows_per_thread;
                int row_end = (t == opt.num_threads - 1) ? outh : (t + 1) * rows_per_thread;
                
                for (int i = row_start; i < row_end; i++)
                {
                    int src_row = i / repeat_h;
                    const float* src_ptr = ptr + src_row * bottom_blob.w;
                    float* dst_ptr = outptr + i * outw;
                    
                    // Copy row with NEON
                    const int nn = bottom_blob.w >> 2;
                    const int remain = bottom_blob.w - (nn << 2);
                    
                    for (int j = 0; j < nn; j++)
                    {
                        float32x4_t v = vld1q_f32(src_ptr + j * 4);
                        vst1q_f32(dst_ptr + j * 4, v);
                    }
                    for (int j = nn << 2; j < bottom_blob.w; j++)
                    {
                        dst_ptr[j] = src_ptr[j];
                    }
                }
            }
            return 0;
        }
        #endif

        // General path with OpenMP
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

    // Legacy mode: use parameters (unchanged)
    const Mat& bottom_blob = bottom_blobs[0];
    int dims = bottom_blob.dims;
    int repeat_w = 1;
    int repeat_h = 1;
    int repeat_d = 1;
    int repeat_c = 1;

    const int repeats_num = repeats.w;

    if (repeats.empty())
    {
        if (dims == 1)
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
        const int* repeats_ptr = repeats;
        if (repeats_num == 1) repeat_w = repeats_ptr[0];
        if (repeats_num == 2) { repeat_h = repeats_ptr[0]; repeat_w = repeats_ptr[1]; }
        if (repeats_num == 3) { repeat_c = repeats_ptr[0]; repeat_h = repeats_ptr[1]; repeat_w = repeats_ptr[2]; }
        if (repeats_num == 4) { repeat_c = repeats_ptr[0]; repeat_d = repeats_ptr[1]; repeat_h = repeats_ptr[2]; repeat_w = repeats_ptr[3]; }
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
