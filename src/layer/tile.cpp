// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "tile.h"

namespace ncnn {

Tile::Tile()
{
    one_blob_only = false;  // Changed to support ONNX mode with 2 inputs
    support_inplace = false;
    axis = 0;
    tiles = 1;
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
        // Use w for 1D tensor, total() can be unreliable for int32 tensors
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
        
        for (int q = 0; q < outc; q++)
        {
            const float* ptr_channel = ptr + bottom_blob.cstep * (q / repeat_c);
            float* outptr_channel = outptr + top_blob.cstep * q;
            
            for (int i = 0; i < outh; i++)
            {
                const float* ptr_row = ptr_channel + bottom_blob.w * (i / repeat_h);
                float* outptr_row = outptr_channel + top_blob.w * i;
                
                for (int j = 0; j < outw; j++)
                {
                    outptr_row[j] = ptr_row[j / repeat_w];
                }
            }
        }
        
        return 0;
    }
    
    // Legacy mode: use parameters
    const Mat& bottom_blob = bottom_blobs[0];
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
            repeat_c = repeats_ptr[0];
            repeat_h = repeats_ptr[1];
            repeat_w = repeats_ptr[2];
        }
        if (repeats_num == 4)
        {
            repeat_c = repeats_ptr[0];
            repeat_d = repeats_ptr[1];
            repeat_h = repeats_ptr[2];
            repeat_w = repeats_ptr[3];
        }
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

    for (int q = 0; q < outc; q++)
    {
        const float* ptr_channel = ptr + bottom_blob.cstep * (q / repeat_c);
        float* outptr_channel = outptr + top_blob.cstep * q;

        for (int i = 0; i < outh; i++)
        {
            const float* ptr_row = ptr_channel + bottom_blob.w * (i / repeat_h);
            float* outptr_row = outptr_channel + top_blob.w * i;

            for (int j = 0; j < outw; j++)
            {
                outptr_row[j] = ptr_row[j / repeat_w];
            }
        }
    }

    return 0;
}

} // namespace ncnn
