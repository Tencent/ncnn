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

int main()
{
    return test_mat_from_bfloat16() || test_modelbin_bfloat16();
}
