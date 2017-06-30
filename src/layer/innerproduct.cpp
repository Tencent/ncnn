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

#include "innerproduct.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(InnerProduct)

InnerProduct::InnerProduct()
{
    one_blob_only = true;
    support_inplace = false;
}

InnerProduct::~InnerProduct()
{
}

#if NCNN_STDIO
#if NCNN_STRING
int InnerProduct::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d %d %d",
                       &num_output, &bias_term, &weight_data_size);
    if (nscan != 3)
    {
        fprintf(stderr, "InnerProduct load_param failed %d\n", nscan);
        return -1;
    }

    return 0;
}
#endif // NCNN_STRING
int InnerProduct::load_param_bin(FILE* paramfp)
{
    fread(&num_output, sizeof(int), 1, paramfp);

    fread(&bias_term, sizeof(int), 1, paramfp);

    fread(&weight_data_size, sizeof(int), 1, paramfp);

    return 0;
}

int InnerProduct::load_model(FILE* binfp)
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
        fprintf(stderr, "InnerProduct read flag_struct failed %d\n", nread);
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
            fprintf(stderr, "InnerProduct read float16_weights failed %d\n", nread);
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
            fprintf(stderr, "InnerProduct read quantization_value failed %d\n", nread);
            return -1;
        }

        int align_weight_data_size = alignSize(weight_data_size * sizeof(unsigned char), 4);
        std::vector<unsigned char> index_array;
        index_array.resize(align_weight_data_size);
        nread = fread(index_array.data(), align_weight_data_size, 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "InnerProduct read index_array failed %d\n", nread);
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
            fprintf(stderr, "InnerProduct read weight_data failed %d\n", nread);
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
            fprintf(stderr, "InnerProduct read bias_data failed %d\n", nread);
            return -1;
        }
    }

    return 0;
}
#endif // NCNN_STDIO

int InnerProduct::load_param(const unsigned char*& mem)
{
    num_output = *(int*)(mem);
    mem += 4;

    bias_term = *(int*)(mem);
    mem += 4;

    weight_data_size = *(int*)(mem);
    mem += 4;

    return 0;
}

int InnerProduct::load_model(const unsigned char*& mem)
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

int InnerProduct::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    top_blob.create(1, 1, num_output);
    if (top_blob.empty())
        return -100;

    // num_output
    const float* weight_data_ptr = weight_data;
    #pragma omp parallel for
    for (int p=0; p<num_output; p++)
    {
        float* outptr = top_blob.channel(p);
        float sum = 0.f;

        if (bias_term)
            sum = bias_data.data[p];

        // channels
        for (int q=0; q<channels; q++)
        {
            const float* w = weight_data_ptr + size * channels * p + size * q;
            const float* m = bottom_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                sum += m[i] * w[i];
            }
        }

        outptr[0] = sum;
    }

    return 0;
}

} // namespace ncnn
