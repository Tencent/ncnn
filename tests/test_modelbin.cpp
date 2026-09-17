// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <stdio.h>

#include "datareader.h"
#include "modelbin.h"

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

int main()
{
    return 0
           || test_mat_from_bfloat16()
           || test_modelbin_bfloat16()
           || test_modelbin_int8();
}
