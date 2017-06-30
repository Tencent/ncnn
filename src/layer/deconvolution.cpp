// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "deconvolution.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Deconvolution)

Deconvolution::Deconvolution()
{
    one_blob_only = true;
    support_inplace = false;
}

Deconvolution::~Deconvolution()
{
}

#if NCNN_STDIO
#if NCNN_STRING
int Deconvolution::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d %d %d %d %d %d %d",
                       &num_output, &kernel_size, &dilation, &stride, &pad, &bias_term,
                       &weight_data_size);
    if (nscan != 7)
    {
        fprintf(stderr, "Deconvolution load_param failed %d\n", nscan);
        return -1;
    }

    return 0;
}
#endif // NCNN_STRING
int Deconvolution::load_param_bin(FILE* paramfp)
{
    fread(&num_output, sizeof(int), 1, paramfp);

    fread(&kernel_size, sizeof(int), 1, paramfp);

    fread(&dilation, sizeof(int), 1, paramfp);

    fread(&stride, sizeof(int), 1, paramfp);

    fread(&pad, sizeof(int), 1, paramfp);

    fread(&bias_term, sizeof(int), 1, paramfp);

    fread(&weight_data_size, sizeof(int), 1, paramfp);

    return 0;
}

int Deconvolution::load_model(FILE* binfp)
{
    int nread;

    union
    {
        struct
        {
            unsigned char f0;
            unsigned char f1;
            unsigned char f2;
            unsigned char f3;
        };
        unsigned int tag;
    } flag_struct;

    nread = fread(&flag_struct, sizeof(flag_struct), 1, binfp);
    if (nread != 1)
    {
        fprintf(stderr, "Deconvolution read flag_struct failed %d\n", nread);
        return -1;
    }

    unsigned int flag = flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;

    weight_data.create(weight_data_size);
    if (weight_data.empty())
        return -100;

    if (flag_struct.tag == 0x01306B47)
    {
        // half-precision weight data
        int align_weight_data_size = alignSize(weight_data_size * sizeof(unsigned short), 4);
        std::vector<unsigned short> float16_weights;
        float16_weights.resize(align_weight_data_size);
        nread = fread(float16_weights.data(), align_weight_data_size, 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "Deconvolution read float16_weights failed %d\n", nread);
            return -1;
        }

        weight_data = Mat::from_float16(float16_weights.data(), weight_data_size);
        if (weight_data.empty())
            return -100;
    }
    else if (flag != 0)
    {
        // quantized weight data
        float quantization_value[256];
        nread = fread(quantization_value, 256 * sizeof(float), 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "Deconvolution read quantization_value failed %d\n", nread);
            return -1;
        }

        int align_weight_data_size = alignSize(weight_data_size * sizeof(unsigned char), 4);
        std::vector<unsigned char> index_array;
        index_array.resize(align_weight_data_size);
        nread = fread(index_array.data(), align_weight_data_size, 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "Deconvolution read index_array failed %d\n", nread);
            return -1;
        }

        float* weight_data_ptr = weight_data;
        for (int i = 0; i < weight_data_size; i++)
        {
            weight_data_ptr[i] = quantization_value[ index_array[i] ];
        }
    }
    else if (flag_struct.f0 == 0)
    {
        // raw weight data
        nread = fread(weight_data, weight_data_size * sizeof(float), 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "Deconvolution read weight_data failed %d\n", nread);
            return -1;
        }
    }

    if (bias_term)
    {
        bias_data.create(num_output);
        if (bias_data.empty())
            return -100;
        nread = fread(bias_data, num_output * sizeof(float), 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "Deconvolution read bias_data failed %d\n", nread);
            return -1;
        }
    }

    return 0;
}
#endif // NCNN_STDIO

int Deconvolution::load_param(const unsigned char*& mem)
{
    num_output = *(int*)(mem);
    mem += 4;

    kernel_size = *(int*)(mem);
    mem += 4;

    dilation = *(int*)(mem);
    mem += 4;

    stride = *(int*)(mem);
    mem += 4;

    pad = *(int*)(mem);
    mem += 4;

    bias_term = *(int*)(mem);
    mem += 4;

    weight_data_size = *(int*)(mem);
    mem += 4;

    return 0;
}

int Deconvolution::load_model(const unsigned char*& mem)
{
    union
    {
        struct
        {
            unsigned char f0;
            unsigned char f1;
            unsigned char f2;
            unsigned char f3;
        };
        unsigned int tag;
    } flag_struct;

    memcpy(&flag_struct, mem, sizeof(flag_struct));
    mem += sizeof(flag_struct);

    unsigned int flag = flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;

    if (flag_struct.tag == 0x01306B47)
    {
        // half-precision weight data
        weight_data = Mat::from_float16((unsigned short*)mem, weight_data_size);
        mem += alignSize(weight_data_size * sizeof(unsigned short), 4);
        if (weight_data.empty())
            return -100;
    }
    else if (flag != 0)
    {
        // quantized weight data
        const float* quantization_value = (const float*)mem;
        mem += 256 * sizeof(float);

        const unsigned char* index_array = (const unsigned char*)mem;
        mem += alignSize(weight_data_size * sizeof(unsigned char), 4);

        weight_data.create(weight_data_size);
        if (weight_data.empty())
            return -100;
        float* weight_data_ptr = weight_data;
        for (int i = 0; i < weight_data_size; i++)
        {
            weight_data_ptr[i] = quantization_value[ index_array[i] ];
        }
    }
    else if (flag_struct.f0 == 0)
    {
        // raw weight data
        weight_data = Mat(weight_data_size, (float*)mem);
        mem += weight_data_size * sizeof(float);
    }

    if (bias_term)
    {
        bias_data = Mat(num_output, (float*)mem);
        mem += num_output * sizeof(float);
    }

    return 0;
}

int Deconvolution::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    // backward strided convolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

//     fprintf(stderr, "Deconvolution input %d x %d  pad = %d  ksize=%d  stride=%d\n", w, h, pad, kernel_size, stride);

    const int kernel_extent = dilation * (kernel_size - 1) + 1;

    int outw = (w - 1) * stride + kernel_extent;
    int outh = (h - 1) * stride + kernel_extent;

    Mat top_blob_bordered;
    top_blob_bordered.create(outw, outh, num_output);
    if (top_blob_bordered.empty())
        return -100;

    const int maxk = kernel_size * kernel_size;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = outw * dilation - kernel_extent;
        for (int i = 0; i < kernel_size; i++)
        {
            for (int j = 0; j < kernel_size; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation;
            }
            p2 += gap;
        }
    }

    // num_output
    const float* weight_data_ptr = weight_data;
    #pragma omp parallel for
    for (int p=0; p<num_output; p++)
    {
        Mat out = top_blob_bordered.channel(p);

        const float bias = bias_term ? bias_data.data[p] : 0.f;

        out.fill(bias);

        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                float* outptr = out.data + out.w * i*stride + j*stride;

                const float* kptr = weight_data_ptr + maxk * channels * p;

                // channels
                for (int q=0; q<channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    float val = *(m.data + m.w * i + j);

                    for (int k = 0; k < maxk; k++)
                    {
                        float w = kptr[k];
                        outptr[ space_ofs[k] ] += val * w;
                    }

                    kptr += maxk;
                }
            }
        }
    }

    top_blob = top_blob_bordered;

    if (pad > 0)
    {
        copy_cut_border(top_blob_bordered, top_blob, pad, pad, pad, pad);
        if (top_blob.empty())
            return -100;

        outw = top_blob.w;
        outh = top_blob.h;
    }

    return 0;
}

} // namespace ncnn
