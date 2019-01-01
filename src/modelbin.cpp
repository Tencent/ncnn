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

#include "modelbin.h"

#include <stdio.h>
#include <string.h>
#include <vector>
#include "platform.h"

namespace ncnn {

Mat ModelBin::load(int w, int h, int type) const
{
    Mat m = load(w * h, type);
    if (m.empty())
        return m;

    return m.reshape(w, h);
}

Mat ModelBin::load(int w, int h, int c, int type) const
{
    Mat m = load(w * h * c, type);
    if (m.empty())
        return m;

    return m.reshape(w, h, c);
}

#if NCNN_STDIO
ModelBinFromStdio::ModelBinFromStdio(FILE* _binfp) : binfp(_binfp)
{
}

Mat ModelBinFromStdio::load(int w, int type) const
{
    if (!binfp)
        return Mat();

    if (type == 0)
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
            fprintf(stderr, "ModelBin read flag_struct failed %d\n", nread);
            return Mat();
        }

        unsigned int flag = flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;

        if (flag_struct.tag == 0x01306B47)
        {
            // half-precision data
            int align_data_size = alignSize(w * sizeof(unsigned short), 4);
            std::vector<unsigned short> float16_weights;
            float16_weights.resize(align_data_size);
            nread = fread(float16_weights.data(), align_data_size, 1, binfp);
            if (nread != 1)
            {
                fprintf(stderr, "ModelBin read float16_weights failed %d\n", nread);
                return Mat();
            }

            return Mat::from_float16(float16_weights.data(), w);
        }
        else if (flag_struct.tag == 0x000D4B38)
        {
            // int8 data
            int align_data_size = alignSize(w, 4);
            std::vector<signed char> int8_weights;
            int8_weights.resize(align_data_size);
            nread = fread(int8_weights.data(), align_data_size, 1, binfp);
            if (nread != 1)
            {
                fprintf(stderr, "ModelBin read int8_weights failed %d\n", nread);
                return Mat();
            }

            Mat m(w, (size_t)1u);
            if (m.empty())
                return m;

            memcpy(m.data, int8_weights.data(), w);

            return m;
        }
        else if (flag_struct.tag == 0x0002C056)
        {
            Mat m(w);
            if (m.empty())
                return m;

            // raw data with extra scaling
            nread = fread(m, w * sizeof(float), 1, binfp);
            if (nread != 1)
            {
                fprintf(stderr, "ModelBin read weight_data failed %d\n", nread);
                return Mat();
            }

            return m;
        }

        Mat m(w);
        if (m.empty())
            return m;

        if (flag != 0)
        {
            // quantized data
            float quantization_value[256];
            nread = fread(quantization_value, 256 * sizeof(float), 1, binfp);
            if (nread != 1)
            {
                fprintf(stderr, "ModelBin read quantization_value failed %d\n", nread);
                return Mat();
            }

            int align_weight_data_size = alignSize(w * sizeof(unsigned char), 4);
            std::vector<unsigned char> index_array;
            index_array.resize(align_weight_data_size);
            nread = fread(index_array.data(), align_weight_data_size, 1, binfp);
            if (nread != 1)
            {
                fprintf(stderr, "ModelBin read index_array failed %d\n", nread);
                return Mat();
            }

            float* ptr = m;
            for (int i = 0; i < w; i++)
            {
                ptr[i] = quantization_value[ index_array[i] ];
            }
        }
        else if (flag_struct.f0 == 0)
        {
            // raw data
            nread = fread(m, w * sizeof(float), 1, binfp);
            if (nread != 1)
            {
                fprintf(stderr, "ModelBin read weight_data failed %d\n", nread);
                return Mat();
            }
        }

        return m;
    }
    else if (type == 1)
    {
        Mat m(w);
        if (m.empty())
            return m;

        // raw data
        int nread = fread(m, w * sizeof(float), 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "ModelBin read weight_data failed %d\n", nread);
            return Mat();
        }

        return m;
    }
    else
    {
        fprintf(stderr, "ModelBin load type %d not implemented\n", type);
        return Mat();
    }

    return Mat();
}
#endif // NCNN_STDIO

ModelBinFromMemory::ModelBinFromMemory(const unsigned char*& _mem) : mem(_mem)
{
}

Mat ModelBinFromMemory::load(int w, int type) const
{
    if (!mem)
        return Mat();

    if (type == 0)
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
            // half-precision data
            Mat m = Mat::from_float16((unsigned short*)mem, w);
            mem += alignSize(w * sizeof(unsigned short), 4);
            return m;
        }
        else if (flag_struct.tag == 0x000D4B38)
        {
            // int8 data
            Mat m = Mat(w, (signed char*)mem, 1u);
            mem += alignSize(w, 4);
            return m;
        }
        else if (flag_struct.tag == 0x0002C056)
        {
            // raw data with extra scaling
            Mat m = Mat(w, (float*)mem);
            mem += w * sizeof(float);
            return m;
        }

        if (flag != 0)
        {
            // quantized data
            const float* quantization_value = (const float*)mem;
            mem += 256 * sizeof(float);

            const unsigned char* index_array = (const unsigned char*)mem;
            mem += alignSize(w * sizeof(unsigned char), 4);

            Mat m(w);
            if (m.empty())
                return m;

            float* ptr = m;
            for (int i = 0; i < w; i++)
            {
                ptr[i] = quantization_value[ index_array[i] ];
            }

            return m;
        }
        else if (flag_struct.f0 == 0)
        {
            // raw data
            Mat m = Mat(w, (float*)mem);
            mem += w * sizeof(float);
            return m;
        }
    }
    else if (type == 1)
    {
        // raw data
        Mat m = Mat(w, (float*)mem);
        mem += w * sizeof(float);
        return m;
    }
    else
    {
        fprintf(stderr, "ModelBin load type %d not implemented\n", type);
        return Mat();
    }

    return Mat();
}

ModelBinFromMatArray::ModelBinFromMatArray(const Mat* _weights) : weights(_weights)
{
}

Mat ModelBinFromMatArray::load(int /*w*/, int /*type*/) const
{
    if (!weights)
        return Mat();

    Mat m = weights[0];
    weights++;
    return m;
}

} // namespace ncnn
