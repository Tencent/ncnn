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

ModelBin::ModelBin()
{
}

ModelBin::~ModelBin()
{
}

Mat ModelBin::load(int /*w*/, int /*type*/) const
{
    return Mat();
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

Mat ModelBin::load(int w, int h, int d, int c, int type) const
{
    Mat m = load(w * h * d * c, type);
    if (m.empty())
        return m;

    return m.reshape(w, h, d, c);
}

class ModelBinFromDataReaderPrivate
{
public:
    ModelBinFromDataReaderPrivate(const DataReader& _dr)
        : dr(_dr)
    {
    }
    const DataReader& dr;
};

ModelBinFromDataReader::ModelBinFromDataReader(const DataReader& _dr)
    : ModelBin(), d(new ModelBinFromDataReaderPrivate(_dr))
{
}

ModelBinFromDataReader::~ModelBinFromDataReader()
{
    delete d;
}

ModelBinFromDataReader::ModelBinFromDataReader(const ModelBinFromDataReader&)
    : d(0)
{
}

ModelBinFromDataReader& ModelBinFromDataReader::operator=(const ModelBinFromDataReader&)
{
    return *this;
}

Mat ModelBinFromDataReader::load(int w, int type) const
{
    Mat m;

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

        nread = d->dr.read(&flag_struct, sizeof(flag_struct));
        if (nread != sizeof(flag_struct))
        {
            NCNN_LOGE("ModelBin read flag_struct failed %zd", nread);
            return Mat();
        }

#if __BIG_ENDIAN__
        swap_endianness_32(&flag_struct.tag);
#endif

        unsigned int flag = (int)flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;

        if (flag_struct.tag == 0x01306B47)
        {
            // half-precision data
            size_t align_data_size = alignSize(w * sizeof(unsigned short), 4);

#if !__BIG_ENDIAN__
            // try reference data
            const void* refbuf = 0;
            nread = d->dr.reference(align_data_size, &refbuf);
            if (nread == align_data_size)
            {
                m = Mat::from_float16((const unsigned short*)refbuf, w);
            }
            else
#endif
            {
                std::vector<unsigned short> float16_weights;
                float16_weights.resize(align_data_size);
                nread = d->dr.read(&float16_weights[0], align_data_size);
                if (nread != align_data_size)
                {
                    NCNN_LOGE("ModelBin read float16_weights failed %zd", nread);
                    return Mat();
                }

#if __BIG_ENDIAN__
                for (int i = 0; i < w; i++)
                {
                    swap_endianness_16(&float16_weights[i]);
                }
#endif

                m = Mat::from_float16(&float16_weights[0], w);
            }

            return m;
        }
        else if (flag_struct.tag == 0x000D4B38)
        {
            // int8 data
            size_t align_data_size = alignSize(w, 4);

#if !__BIG_ENDIAN__
            // try reference data
            const void* refbuf = 0;
            nread = d->dr.reference(align_data_size, &refbuf);
            if (nread == align_data_size)
            {
                m = Mat(w, (void*)refbuf, (size_t)1u);
            }
            else
#endif
            {
                std::vector<signed char> int8_weights;
                int8_weights.resize(align_data_size);
                nread = d->dr.read(&int8_weights[0], align_data_size);
                if (nread != align_data_size)
                {
                    NCNN_LOGE("ModelBin read int8_weights failed %zd", nread);
                    return Mat();
                }

                m.create(w, (size_t)1u);
                if (m.empty())
                    return m;

                memcpy(m.data, &int8_weights[0], w);
            }

            return m;
        }
        else if (flag_struct.tag == 0x0002C056)
        {
#if !__BIG_ENDIAN__
            // try reference data
            const void* refbuf = 0;
            nread = d->dr.reference(w * sizeof(float), &refbuf);
            if (nread == w * sizeof(float))
            {
                m = Mat(w, (void*)refbuf);
            }
            else
#endif
            {
                m.create(w);
                if (m.empty())
                    return m;

                // raw data with extra scaling
                nread = d->dr.read(m, w * sizeof(float));
                if (nread != w * sizeof(float))
                {
                    NCNN_LOGE("ModelBin read weight_data failed %zd", nread);
                    return Mat();
                }

#if __BIG_ENDIAN__
                for (int i = 0; i < w; i++)
                {
                    swap_endianness_32((float*)m + i);
                }
#endif
            }

            return m;
        }

        if (flag != 0)
        {
            m.create(w);
            if (m.empty())
                return m;

            // quantized data
            float quantization_value[256];
            nread = d->dr.read(quantization_value, 256 * sizeof(float));
            if (nread != 256 * sizeof(float))
            {
                NCNN_LOGE("ModelBin read quantization_value failed %zd", nread);
                return Mat();
            }

#if __BIG_ENDIAN__
            for (int i = 0; i < 256; i++)
            {
                swap_endianness_32(&quantization_value[i]);
            }
#endif

            size_t align_weight_data_size = alignSize(w * sizeof(unsigned char), 4);
            std::vector<unsigned char> index_array;
            index_array.resize(align_weight_data_size);
            nread = d->dr.read(&index_array[0], align_weight_data_size);
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
#if !__BIG_ENDIAN__
            // try reference data
            const void* refbuf = 0;
            nread = d->dr.reference(w * sizeof(float), &refbuf);
            if (nread == w * sizeof(float))
            {
                m = Mat(w, (void*)refbuf);
            }
            else
#endif
            {
                m.create(w);
                if (m.empty())
                    return m;

                // raw data
                nread = d->dr.read(m, w * sizeof(float));
                if (nread != w * sizeof(float))
                {
                    NCNN_LOGE("ModelBin read weight_data failed %zd", nread);
                    return Mat();
                }

#if __BIG_ENDIAN__
                for (int i = 0; i < w; i++)
                {
                    swap_endianness_32((float*)m + i);
                }
#endif
            }
        }

        return m;
    }
    else if (type == 1)
    {
#if !__BIG_ENDIAN__
        // try reference data
        const void* refbuf = 0;
        size_t nread = d->dr.reference(w * sizeof(float), &refbuf);
        if (nread == w * sizeof(float))
        {
            m = Mat(w, (void*)refbuf);
        }
        else
#endif
        {
            m.create(w);
            if (m.empty())
                return m;

            // raw data
            size_t nread = d->dr.read(m, w * sizeof(float));
            if (nread != w * sizeof(float))
            {
                NCNN_LOGE("ModelBin read weight_data failed %zd", nread);
                return Mat();
            }

#if __BIG_ENDIAN__
            for (int i = 0; i < w; i++)
            {
                swap_endianness_32((float*)m + i);
            }
#endif
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

class ModelBinFromMatArrayPrivate
{
public:
    ModelBinFromMatArrayPrivate(const Mat* _weights)
        : weights(_weights)
    {
    }
    mutable const Mat* weights;
};

ModelBinFromMatArray::ModelBinFromMatArray(const Mat* _weights)
    : ModelBin(), d(new ModelBinFromMatArrayPrivate(_weights))
{
}

ModelBinFromMatArray::~ModelBinFromMatArray()
{
    delete d;
}

ModelBinFromMatArray::ModelBinFromMatArray(const ModelBinFromMatArray&)
    : d(0)
{
}

ModelBinFromMatArray& ModelBinFromMatArray::operator=(const ModelBinFromMatArray&)
{
    return *this;
}

Mat ModelBinFromMatArray::load(int /*w*/, int /*type*/) const
{
    if (!d->weights)
        return Mat();

    Mat m = d->weights[0];
    d->weights++;
    return m;
}

} // namespace ncnn
