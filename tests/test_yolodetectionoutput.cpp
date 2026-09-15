// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <float.h>

static int test_yolodetectionoutput_load_param()
{
    ncnn::ParamDict pd;
    pd.set(0, 20);
    pd.set(1, 1);
    ncnn::Mat biases(2);
    biases.fill(1.f);
    pd.set(4, biases);
    if (test_layer_param(ncnn::LayerType::YoloDetectionOutput, pd, 0) != 0)
        return -1;

    const ncnn::ParamDict base = pd;

    if (test_layer_param(ncnn::LayerType::YoloDetectionOutput, base, 4, ncnn::Mat(0), -1) != 0)
        return -1;

    ncnn::Mat missing_data(0);
    missing_data.w = 1;

    return 0
           || test_layer_param(ncnn::LayerType::YoloDetectionOutput, base, 4, missing_data, -1)
           || test_layer_param(ncnn::LayerType::YoloDetectionOutput, base, 4, biases.range(0, 1), -1);
}

static int test_yolodetectionoutput_load_param_biases()
{
    ncnn::ParamDict base;
    base.set(0, 1);
    base.set(1, 1);
    ncnn::Mat biases(3);
    biases.fill(1.f);

    const float valid[] = {0.5f, 1.f, 32.f, FLT_MAX};
    for (int i = 0; i < 4; i++)
    {
        biases.fill(valid[i]);
        if (test_layer_param(ncnn::LayerType::YoloDetectionOutput, base, 4, biases, 0) != 0)
            return -1;
    }
    biases.fill(1.f);

    // only the first num_box pairs are used
    const unsigned int invalid[] = {0x00000000u, 0x80000000u, 0xbf800000u, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0x7f800001u};
    for (int i = 0; i < 7; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            memcpy((float*)biases + j, &invalid[i], sizeof(float));
            if (test_layer_param(ncnn::LayerType::YoloDetectionOutput, base, 4, biases, j < 2 ? -1 : 0) != 0)
            {
                fprintf(stderr, "test_yolodetectionoutput_load_param_biases failed biases[%d]=0x%08x\n", j, invalid[i]);
                return -1;
            }
            biases[j] = 1.f;
        }
    }

    return 0;
}

static int test_yolodetectionoutput_load_param_text()
{
#if NCNN_STRING
    const char* params[] = {
        "0=1 1=1 -23304=2,1,2",
        "0=1 1=1 -23304=2,0.5,1.5",
        "0=1 1=1 -23304=2,2147483647,2147483647",
        "0=1 1=1 -23304=3,1,2,-1",
        "0=1 1=1 -23304=2,0,2",
        "0=1 1=1 -23304=2,1,-1"
    };
    for (int i = 0; i < 6; i++)
    {
        TestParamDict pd;
        if (pd.load_param(params[i]) != 0)
            return -1;

        if (test_layer_param(ncnn::LayerType::YoloDetectionOutput, pd, i < 4 ? 0 : -1) != 0)
        {
            fprintf(stderr, "test_yolodetectionoutput_load_param_text failed params=%s\n", params[i]);
            return -1;
        }
    }
#endif
    return 0;
}

static int test_yolodetectionoutput_load_param_thresholds()
{
    ncnn::ParamDict base;
    base.set(0, 1);
    base.set(1, 1);
    ncnn::Mat biases(2);
    biases.fill(1.f);
    base.set(4, biases);

    const float valid[] = {-1.f, 0.f, 0.5f, 1.f, 2.f};
    const unsigned int special[] = {0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00000u, 0x7f800001u, 0xff800001u};
    for (int id = 2; id <= 3; id++)
    {
        for (int i = 0; i < 5; i++)
        {
            if (test_layer_param(ncnn::LayerType::YoloDetectionOutput, base, id, valid[i], 0) != 0)
                return -1;
        }

        for (int i = 0; i < 6; i++)
        {
            float value;
            memcpy(&value, &special[i], sizeof(value));
            const int expected_ret = i < 2 ? 0 : -1;
            if (test_layer_param(ncnn::LayerType::YoloDetectionOutput, base, id, value, expected_ret) != 0)
                return -1;

            // binary scalar parameters have no integer or float type tag
            unsigned char binary[] = {0, 0, 0, 0, 0, 0, 0, 0, 0x17, 0xff, 0xff, 0xff};
            binary[0] = (unsigned char)id;
            for (int j = 0; j < 4; j++)
                binary[4 + j] = (unsigned char)(special[i] >> (j * 8));

            TestParamDict pd;
            if (pd.load_param_bin(binary) != 0)
                return -1;
            pd.set(0, 1);
            pd.set(1, 1);
            pd.set(4, biases);
            if (test_layer_param(ncnn::LayerType::YoloDetectionOutput, pd, expected_ret) != 0)
                return -1;
        }
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_yolodetectionoutput_load_param()
           || test_yolodetectionoutput_load_param_biases()
           || test_yolodetectionoutput_load_param_text()
           || test_yolodetectionoutput_load_param_thresholds();
}
