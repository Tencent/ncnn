// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <limits.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#include "datareader.h"
#include "modelbin.h"
#include "layer.h"
#include "paramdict.h"

static int test_mat_from_bfloat16()
{
    const unsigned short data[] = {0x3f80, 0xc020, 0x3f00};
    ncnn::Mat m = ncnn::Mat::from_bfloat16(data, 3);

    if (m.empty() || m.elemsize != 4u || m[0] != 1.f || m[1] != -2.5f || m[2] != 0.5f)
    {
        fprintf(stderr, "test_mat_from_bfloat16 failed\n");
        return -1;
    }

    return 0;
}

static int test_modelbin_bfloat16()
{
    const unsigned char model_data[] = {
        0x83, 0x8b, 0x34, 0x01,
        0x80, 0x3f, 0x20, 0xc0, 0x00, 0x3f, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x80, 0x40
    };

    const unsigned char* mem = model_data;
    ncnn::DataReaderFromMemory dr(mem);
    ncnn::ModelBinFromDataReader mb(dr);

    ncnn::Mat m = mb.load(3, 0);
    if (m.empty() || m.elemsize != 4u || m[0] != 1.f || m[1] != -2.5f || m[2] != 0.5f)
    {
        fprintf(stderr, "test_modelbin_bfloat16 failed\n");
        return -1;
    }

    ncnn::Mat m2 = mb.load(1, 0);
    if (m2.empty() || m2[0] != 4.f)
    {
        fprintf(stderr, "test_modelbin_bfloat16 alignment failed\n");
        return -1;
    }

    return 0;
}

class DataReaderFromMemoryNoReference : public ncnn::DataReader
{
public:
    DataReaderFromMemoryNoReference(const unsigned char* _data, size_t _size)
        : data(_data), size(_size), offset(0)
    {
    }

    virtual size_t read(void* buf, size_t read_size) const
    {
        if (offset >= size)
            return 0;

        size_t remain = size - offset;
        if (read_size > remain)
            read_size = remain;

        memcpy(buf, data + offset, read_size);
        offset += read_size;
        return read_size;
    }

private:
    const unsigned char* data;
    size_t size;
    mutable size_t offset;
};

static int test_modelbin_int8(const ncnn::DataReader& dr, int w)
{
    ncnn::ModelBinFromDataReader mb(dr);

    ncnn::Mat m = mb.load(w, 3);
    if (m.empty() || m.w != w || m.elemsize != 1u || m.elempack != 1)
    {
        fprintf(stderr, "test_modelbin_int8 shape failed w=%d\n", w);
        return -1;
    }

    const signed char expected[] = {0, 1, -128, 127};
    const signed char* p = m;
    for (int i = 0; i < w; i++)
    {
        if (p[i] != expected[i])
        {
            fprintf(stderr, "test_modelbin_int8 value failed w=%d i=%d %d != %d\n", w, i, p[i], expected[i]);
            return -1;
        }
    }

    ncnn::Mat m2 = mb.load(1, 1);
    if (m2.empty() || m2[0] != 4.f)
    {
        fprintf(stderr, "test_modelbin_int8 alignment failed w=%d\n", w);
        return -1;
    }

    return 0;
}

static int test_modelbin_int8(int w)
{
    unsigned char model_data[] = {
        0x00, 0x01, 0x80, 0x7f,
        0x00, 0x00, 0x80, 0x40
    };
    for (int i = w; i < 4; i++)
        model_data[i] = 0;

    // reference path
    const unsigned char* mem = model_data;
    ncnn::DataReaderFromMemory dr(mem);
    int ret = test_modelbin_int8(dr, w);
    if (ret != 0)
        return ret;

    // copy path
    // DataReaderFromMemoryNoReference inherits DataReader::reference(),
    // which returns 0, so ModelBinFromDataReader falls back to read().
    DataReaderFromMemoryNoReference dr2(model_data, sizeof(model_data));
    ret = test_modelbin_int8(dr2, w);
    if (ret != 0)
        return ret;

    return 0;
}

static int test_modelbin_int8_short_read()
{
    // three int8 elements require four bytes in the model because raw
    // int8 data is padded to a 32-bit boundary
    const unsigned char model_data[] = {
        0x00, 0x01, 0x00
    };

    DataReaderFromMemoryNoReference dr(model_data, sizeof(model_data));
    ncnn::ModelBinFromDataReader mb(dr);

    ncnn::Mat m = mb.load(3, 3);
    if (!m.empty())
    {
        fprintf(stderr, "test_modelbin_int8_short_read failed\n");
        return -1;
    }

    return 0;
}

static int test_modelbin_int8()
{
    return 0
           || test_modelbin_int8(1)
           || test_modelbin_int8(2)
           || test_modelbin_int8(3)
           || test_modelbin_int8(4)
           || test_modelbin_int8_short_read();
}

static int test_modelbin_bounded()
{
    const unsigned char model_data[][16] = {
        {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x20, 0xc0, 0x00, 0x00, 0x00, 0x3f},
        {0x47, 0x6b, 0x30, 0x01, 0x00, 0x3c, 0x00, 0xc1, 0x00, 0x38, 0x00, 0x00},
        {0x83, 0x8b, 0x34, 0x01, 0x80, 0x3f, 0x20, 0xc0, 0x00, 0x3f, 0x00, 0x00},
        {0x38, 0x4b, 0x0d, 0x00, 0x00, 0x01, 0x80, 0x00},
        {0x00, 0x01, 0x80, 0x00},
        {0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x20, 0xc0, 0x00, 0x00, 0x00, 0x3f}
    };
    const size_t sizes[] = {16, 12, 12, 8, 4, 12};
    const int types[] = {0, 0, 0, 0, 3, 1};

    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++)
    {
        // truncate at every byte, including format tags and alignment padding
        for (size_t size = 0; size <= sizes[i]; size++)
        {
            const unsigned char* mem = model_data[i];
            ncnn::DataReaderFromMemory dr(mem, size);
            ncnn::ModelBinFromDataReader mb(dr);
            ncnn::Mat m = mb.load(3, types[i]);
            if (m.empty() != (size < sizes[i]) || mem != model_data[i] + size)
            {
                fprintf(stderr, "test_modelbin_bounded failed format=%zu size=%zu\n", i, size);
                return -1;
            }

            if (size == sizes[i])
            {
                if (i == 3 || i == 4)
                {
                    const signed char* p = m;
                    if (m.elemsize != 1u || p[0] != 0 || p[1] != 1 || p[2] != -128)
                    {
                        fprintf(stderr, "test_modelbin_bounded int8 values failed\n");
                        return -1;
                    }
                }
                else if (m.elemsize != 4u || m[0] != 1.f || m[1] != -2.5f || m[2] != 0.5f)
                {
                    fprintf(stderr, "test_modelbin_bounded float values failed format=%zu\n", i);
                    return -1;
                }
            }
        }
    }

    return 0;
}

// the element count of a multi-dimensional load must not wrap or come from negative factors
static int test_modelbin_invalid_shape()
{
    // raw float tag followed by 131073 floats, enough for the wrapped products below
    std::vector<unsigned char> model_data(4 + 131073 * 4, 0);

    struct Case
    {
        int dims;
        int w, h, d, c;
    };
    const Case cases[] = {
        {2, 65537, 65537, 1, 1},
        {3, 65537, 65537, 1, 1},
        {4, 65537, 65537, 1, 1},
        {3, 1, 65537, 1, 65537},
        {4, 1, 1, 65537, 65537},
        {2, -4, -4, 1, 1},
        {3, -1, 4, 1, -4},
        {4, 2, -2, 2, -2},
        {3, 4, 4, 1, -4},
        {4, 2, 2, 2, -2},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        const Case& s = cases[i];
        const unsigned char* mem = &model_data[0];
        ncnn::DataReaderFromMemory dr(mem, model_data.size());
        ncnn::ModelBinFromDataReader mbdr(dr);
        const ncnn::ModelBin& mb = mbdr;
        ncnn::Mat m;
        if (s.dims == 2)
            m = mb.load(s.w, s.h, 0);
        else if (s.dims == 3)
            m = mb.load(s.w, s.h, s.c, 0);
        else
            m = mb.load(s.w, s.h, s.d, s.c, 0);
        if (!m.empty())
        {
            fprintf(stderr, "test_modelbin_invalid_shape failed dims=%d %d %d %d %d -> %d %d %d %d\n", s.dims, s.w, s.h, s.d, s.c, m.w, m.h, m.d, m.c);
            return -1;
        }
    }

    // valid shapes still load
    {
        const unsigned char* mem = &model_data[0];
        ncnn::DataReaderFromMemory dr(mem, model_data.size());
        ncnn::ModelBinFromDataReader mbdr(dr);
        const ncnn::ModelBin& mb = mbdr;
        ncnn::Mat m2 = mb.load(4, 4, 0);
        ncnn::Mat m3 = mb.load(2, 2, 2, 0);
        ncnn::Mat m4 = mb.load(2, 2, 2, 2, 0);
        if (m2.empty() || m2.dims != 2 || m2.w != 4 || m2.h != 4
                || m3.empty() || m3.dims != 3 || m3.c != 2
                || m4.empty() || m4.dims != 4 || m4.d != 2 || m4.c != 2)
        {
            fprintf(stderr, "test_modelbin_invalid_shape valid shape failed\n");
            return -1;
        }
    }

    // a negative element count is rejected before any byte-size arithmetic
    const int negative[] = {-1, -4, -1073741824, INT_MIN};
    for (int type = 0; type < 2; type++)
    {
        for (size_t i = 0; i < sizeof(negative) / sizeof(negative[0]); i++)
        {
            const unsigned char* mem = &model_data[0];
            ncnn::DataReaderFromMemory dr(mem, model_data.size());
            ncnn::ModelBinFromDataReader mb(dr);
            ncnn::Mat m = mb.load(negative[i], type);
            if (!m.empty())
            {
                fprintf(stderr, "test_modelbin_invalid_shape negative count failed w=%d type=%d\n", negative[i], type);
                return -1;
            }
        }
    }

    // the same through the fp16 and int8 tagged paths
    const unsigned char tags[][4] = {{0x47, 0x6b, 0x30, 0x01}, {0x38, 0x4b, 0x0d, 0x00}};
    for (int t = 0; t < 2; t++)
    {
        for (size_t i = 0; i < sizeof(negative) / sizeof(negative[0]); i++)
        {
            memcpy(&model_data[0], tags[t], 4);
            const unsigned char* mem = &model_data[0];
            ncnn::DataReaderFromMemory dr(mem, model_data.size());
            ncnn::ModelBinFromDataReader mb(dr);
            ncnn::Mat m = mb.load(negative[i], 0);
            memset(&model_data[0], 0, 4);
            if (!m.empty())
            {
                fprintf(stderr, "test_modelbin_invalid_shape tagged negative count failed w=%d tag=%d\n", negative[i], t);
                return -1;
            }
        }
    }

    return 0;
}

#if NCNN_INT8
static void append_bytes(std::vector<unsigned char>& data, const void* p, size_t size)
{
    const unsigned char* b = (const unsigned char*)p;
    data.insert(data.end(), b, b + size);
}

// int8-tagged weights followed by optional raw float scale records
static std::vector<unsigned char> int8_weights_model(int weight_count, int scale_count)
{
    std::vector<unsigned char> data;
    const unsigned char tag[4] = {0x38, 0x4b, 0x0d, 0x00};
    append_bytes(data, tag, 4);
    for (int i = 0; i < weight_count; i++)
        data.push_back((unsigned char)(i + 1));
    while (data.size() % 4)
        data.push_back(0);
    for (int i = 0; i < scale_count; i++)
    {
        float s = 64.f;
        append_bytes(data, &s, 4);
    }
    return data;
}

static int check_int8_weights_load(const char* type, const ncnn::ParamDict& pd, int weight_count, int scale_count, int expected_ret)
{
    std::vector<unsigned char> data = int8_weights_model(weight_count, scale_count);

    ncnn::Layer* layer = ncnn::create_layer(type);
    if (!layer)
    {
        fprintf(stderr, "test_modelbin_int8_weights create_layer %s failed\n", type);
        return -1;
    }

    int ret = layer->load_param(pd);
    if (ret == 0)
    {
        const unsigned char* mem = &data[0];
        ncnn::DataReaderFromMemory dr(mem, data.size());
        ncnn::ModelBinFromDataReader mb(dr);
        ret = layer->load_model(mb);
    }

    delete layer;

    if (ret != expected_ret)
    {
        fprintf(stderr, "test_modelbin_int8_weights %s failed ret=%d expected=%d\n", type, ret, expected_ret);
        return -1;
    }

    return 0;
}

// int8 weights are only consumable together with their int8 scale records
static int test_modelbin_int8_weights()
{
    for (int scale = 0; scale < 2; scale++)
    {
        {
            ncnn::ParamDict pd;
            pd.set(0, 1);
            pd.set(1, 1);
            pd.set(6, 4);
            pd.set(8, scale);
            if (check_int8_weights_load("Convolution", pd, 4, scale ? 2 : 0, scale ? 0 : -1))
                return -1;
        }
        {
            ncnn::ParamDict pd;
            pd.set(0, 2);
            pd.set(1, 1);
            pd.set(6, 2);
            pd.set(7, 2);
            pd.set(8, scale);
            if (check_int8_weights_load("ConvolutionDepthWise", pd, 2, scale ? 3 : 0, scale ? 0 : -1))
                return -1;
        }
        {
            ncnn::ParamDict pd;
            pd.set(0, 1);
            pd.set(2, 4);
            pd.set(8, scale);
            if (check_int8_weights_load("InnerProduct", pd, 4, scale ? 2 : 0, scale ? 0 : -1))
                return -1;
        }
        {
            ncnn::ParamDict pd;
            pd.set(0, 2);
            pd.set(1, 2);
            pd.set(3, 4);
            pd.set(18, scale ? 2 : 0);
            if (check_int8_weights_load("Embed", pd, 4, scale ? 1 : 0, scale ? 0 : -1))
                return -1;
        }
    }

    // float weights cannot feed the int8 Embed path
    {
        std::vector<unsigned char> data(4 + 4 * 4 + 4, 0);
        float s = 64.f;
        memcpy(&data[20], &s, 4);
        ncnn::ParamDict pd;
        pd.set(0, 2);
        pd.set(1, 2);
        pd.set(3, 4);
        pd.set(18, 2);
        ncnn::Layer* layer = ncnn::create_layer("Embed");
        int ret = layer->load_param(pd);
        if (ret == 0)
        {
            const unsigned char* mem = &data[0];
            ncnn::DataReaderFromMemory dr(mem, data.size());
            ncnn::ModelBinFromDataReader mb(dr);
            ret = layer->load_model(mb);
        }
        delete layer;
        if (ret != -1)
        {
            fprintf(stderr, "test_modelbin_int8_weights Embed float weights with int8_scale_term failed ret=%d\n", ret);
            return -1;
        }
    }

    return 0;
}
#endif // NCNN_INT8

int main()
{
    return 0
           || test_mat_from_bfloat16()
           || test_modelbin_bfloat16()
           || test_modelbin_int8()
           || test_modelbin_bounded()
           || test_modelbin_invalid_shape()
#if NCNN_INT8
           || test_modelbin_int8_weights()
#endif
           ;
}
