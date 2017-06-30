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

#include "embed.h"
#include <string.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(Embed)

Embed::Embed()
{
    one_blob_only = true;
    support_inplace = false;
}

Embed::~Embed()
{
}

#if NCNN_STDIO
#if NCNN_STRING
int Embed::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d %d %d %d",
                       &num_output, &input_dim, &bias_term, &weight_data_size);
    if (nscan != 4)
    {
        fprintf(stderr, "Embed load_param failed %d\n", nscan);
        return -1;
    }

    return 0;
}
#endif // NCNN_STRING
int Embed::load_param_bin(FILE* paramfp)
{
    fread(&num_output, sizeof(int), 1, paramfp);

    fread(&input_dim, sizeof(int), 1, paramfp);

    fread(&bias_term, sizeof(int), 1, paramfp);

    fread(&weight_data_size, sizeof(int), 1, paramfp);

    return 0;
}

int Embed::load_model(FILE* binfp)
{
    int nread;

    struct
    {
        unsigned char f0;
        unsigned char f1;
        unsigned char f2;
        unsigned char f3;
    } flag_struct;

    nread = fread(&flag_struct, sizeof(flag_struct), 1, binfp);
    if (nread != 1)
    {
        fprintf(stderr, "Embed read flag_struct failed %d\n", nread);
        return -1;
    }

    unsigned int flag = flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;

    weight_data.create(weight_data_size);
    if (weight_data.empty())
        return -100;

    if (flag != 0)
    {
        // quantized weight data
        float quantization_value[256];
        nread = fread(quantization_value, 256 * sizeof(float), 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "Embed read quantization_value failed %d\n", nread);
            return -1;
        }

        std::vector<unsigned char> index_array;
        index_array.resize(weight_data_size);
        nread = fread(index_array.data(), weight_data_size * sizeof(unsigned char), 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "Embed read index_array failed %d\n", nread);
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
            fprintf(stderr, "Embed read weight_data failed %d\n", nread);
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
            fprintf(stderr, "Embed read bias_data failed %d\n", nread);
            return -1;
        }
    }

    return 0;
}
#endif // NCNN_STDIO

int Embed::load_param(const unsigned char*& mem)
{
    num_output = *(int*)(mem);
    mem += 4;

    input_dim = *(int*)(mem);
    mem += 4;

    bias_term = *(int*)(mem);
    mem += 4;

    weight_data_size = *(int*)(mem);
    mem += 4;

    return 0;
}

int Embed::load_model(const unsigned char*& mem)
{
    struct
    {
        unsigned char f0;
        unsigned char f1;
        unsigned char f2;
        unsigned char f3;
    } flag_struct;

    memcpy(&flag_struct, mem, sizeof(flag_struct));
    mem += sizeof(flag_struct);

    unsigned int flag = flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;

    if (flag != 0)
    {
        // quantized weight data
        const float* quantization_value = (const float*)mem;
        mem += 256 * sizeof(float);

        const unsigned char* index_array = (const unsigned char*)mem;
        mem += weight_data_size * sizeof(unsigned char);

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

int Embed::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    int words = bottom_blob.total();

    top_blob.create(num_output, words, 1);
    if (top_blob.empty())
        return -100;

    // num_output
    const float* word_ptr = bottom_blob;
    const float* dict_ptr = weight_data;
    #pragma omp parallel for
    for (int q=0; q<words; q++)
    {
        float* outptr = top_blob.data + top_blob.w * q;

        int word_index = (int)word_ptr[q];

        // check word_index >= 0 && word_index < input_dim

        const float* em = dict_ptr + num_output * word_index;

        memcpy(outptr, em, num_output * sizeof(float));

        if (bias_term)
        {
            for (int p=0; p<num_output; p++)
            {
                outptr[p] += bias_data.data[p];
            }
        }
    }

    return 0;
}

} // namespace ncnn
