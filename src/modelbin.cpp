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

#include "datareader.h"

#include <string.h>

namespace ncnn {

ModelBin::~ModelBin()
{
}

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

ModelBinFromDataReader::ModelBinFromDataReader(const DataReader& _dr)
    : dr(_dr)
{
}

Mat ModelBinFromDataReader::load(int w, int type) const
{
    if (type == 0)
    {
        size_t nread;

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

        nread = dr.read(&flag_struct, sizeof(flag_struct));
        if (nread != sizeof(flag_struct))
        {
            NCNN_LOGE("ModelBin read flag_struct failed %zd", nread);
            return Mat();
        }

        unsigned int flag = flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;

        if (flag_struct.tag == 0x01306B47)
        {
            // half-precision data
            size_t align_data_size = alignSize(w * sizeof(unsigned short), 4);
            std::vector<unsigned short> float16_weights;
            float16_weights.resize(align_data_size);
            nread = dr.read(float16_weights.data(), align_data_size);
            if (nread != align_data_size)
            {
                NCNN_LOGE("ModelBin read float16_weights failed %zd", nread);
                return Mat();
            }

            return Mat::from_float16(float16_weights.data(), w);
        }
        else if (flag_struct.tag == 0x000D4B38)
        {
            // int8 data
            size_t align_data_size = alignSize(w, 4);
            std::vector<signed char> int8_weights;
            int8_weights.resize(align_data_size);
            nread = dr.read(int8_weights.data(), align_data_size);
            if (nread != align_data_size)
            {
                NCNN_LOGE("ModelBin read int8_weights failed %zd", nread);
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
            nread = dr.read(m, w * sizeof(float));
            if (nread != w * sizeof(float))
            {
                NCNN_LOGE("ModelBin read weight_data failed %zd", nread);
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
            nread = dr.read(quantization_value, 256 * sizeof(float));
            if (nread != 256 * sizeof(float))
            {
                NCNN_LOGE("ModelBin read quantization_value failed %zd", nread);
                return Mat();
            }

            size_t align_weight_data_size = alignSize(w * sizeof(unsigned char), 4);
            std::vector<unsigned char> index_array;
            index_array.resize(align_weight_data_size);
            nread = dr.read(index_array.data(), align_weight_data_size);
            if (nread != align_weight_data_size)
            {
                NCNN_LOGE("ModelBin read index_array failed %zd", nread);
                return Mat();
            }

            float* ptr = m;
            for (int i = 0; i < w; i++)
            {
                ptr[i] = quantization_value[index_array[i]];
            }
        }
        else if (flag_struct.f0 == 0)
        {
            // raw data
            nread = dr.read(m, w * sizeof(float));
            if (nread != w * sizeof(float))
            {
                NCNN_LOGE("ModelBin read weight_data failed %zd", nread);
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
        size_t nread = dr.read(m, w * sizeof(float));
        if (nread != w * sizeof(float))
        {
            NCNN_LOGE("ModelBin read weight_data failed %zd", nread);
            return Mat();
        }

        return m;
    }
    else
    {
        NCNN_LOGE("ModelBin load type %d not implemented", type);
        return Mat();
    }

    return Mat();
}

ModelBinFromMatArray::ModelBinFromMatArray(const Mat* _weights)
    : weights(_weights)
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
