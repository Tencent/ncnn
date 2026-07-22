// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "net.h"

#if NCNN_STDIO && NCNN_STRING
static int test_external_input_lightmode()
{
    const char param[] = "7767517\n"
                         "2 2\n"
                         "Input input 0 1 in\n"
                         "ReLU relu 1 1 in out\n";

    ncnn::Net net;
    if (net.load_param_mem(param) != 0)
        return -1;

    const float input_expected[] = {-1.f, 2.f, -3.f, 4.f};
    float input_data[4] = {-1.f, 2.f, -3.f, 4.f};
    ncnn::Mat input(4, input_data);

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);

    ncnn::Mat output;
    if (ex.input("in", input) != 0 || ex.extract("out", output) != 0)
        return -1;

    const float expected[] = {0.f, 2.f, 0.f, 4.f};
    const float* output_data = (const float*)output.data;
    for (int i = 0; i < 4; i++)
    {
        if (output_data[i] != expected[i] || input_data[i] != input_expected[i])
            return -1;
    }

    return 0;
}
#endif

int main()
{
#if NCNN_STDIO && NCNN_STRING
    return test_external_input_lightmode();
#else
    return 0;
#endif
}
