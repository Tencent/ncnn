// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include <float.h>

static int test_swiglu(const ncnn::Mat& a, const ncnn::Mat& b, float sigmoid_scale, float up_offset, float activation_limit, int flag = 0)
{
    ncnn::ParamDict pd;
    pd.set(0, sigmoid_scale);
    pd.set(1, up_offset);
    pd.set(2, activation_limit);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(2);
    as[0] = a;
    as[1] = b;

    int ret = test_layer("SwiGLU", pd, weights, as, 1, 0.001f, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_swiglu failed a.dims=%d a=(%d %d %d %d) scale=%f offset=%f limit=%f\n", a.dims, a.w, a.h, a.d, a.c, sigmoid_scale, up_offset, activation_limit);
    }

    return ret;
}

static int test_swiglu_0()
{
    const ncnn::Mat a0 = RandomMat(5, 7, 3, 8);
    const ncnn::Mat b0 = RandomMat(5, 7, 3, 8);
    const ncnn::Mat a1 = RandomMat(5, 7, 3, 7, -12.f, 12.f);
    const ncnn::Mat b1 = RandomMat(5, 7, 3, 7, -12.f, 12.f);
    const ncnn::Mat a2 = RandomMat(4, 9, 2, 16, -12.f, 12.f);
    const ncnn::Mat b2 = RandomMat(4, 9, 2, 16, -12.f, 12.f);
    const ncnn::Mat a3 = RandomMat(3, 5, 7, 5);
    const ncnn::Mat b3 = RandomMat(3, 5, 7, 5);

    // small limits stay within the default random range and exercise the clamp
    return 0
           || test_swiglu(a0, b0, 1.f, 0.f, 0.f)
           || test_swiglu(a0, b0, 1.f, 0.f, 0.5f)
           || test_swiglu(a1, b1, 1.702f, 1.f, 7.f)
           || test_swiglu(a2, b2, 1.f, 0.f, 10.f)
           || test_swiglu(a3, b3, 1.f, 0.5f, 0.f)
           || test_swiglu(a3, b3, 2.f, 0.f, 0.f)
           || test_swiglu(a3, b3, 1.702f, 1.f, 0.5f);
}

static int test_swiglu_1()
{
    const ncnn::Mat a0 = RandomMat(7, 9, 12);
    const ncnn::Mat b0 = RandomMat(7, 9, 12);
    const ncnn::Mat a1 = RandomMat(7, 9, 7, -12.f, 12.f);
    const ncnn::Mat b1 = RandomMat(7, 9, 7, -12.f, 12.f);
    const ncnn::Mat a2 = RandomMat(33, 5, 16, -12.f, 12.f);
    const ncnn::Mat b2 = RandomMat(33, 5, 16, -12.f, 12.f);
    const ncnn::Mat a3 = RandomMat(129, 3, 4);
    const ncnn::Mat b3 = RandomMat(129, 3, 4);

    return 0
           || test_swiglu(a0, b0, 1.f, 0.f, 0.f)
           || test_swiglu(a1, b1, 1.702f, 1.f, 7.f)
           || test_swiglu(a2, b2, 1.f, 0.f, 10.f)
           || test_swiglu(a3, b3, 1.f, 1.f, 0.f)
           || test_swiglu(a3, b3, 1.f, 1.f, 0.5f);
}

static int test_swiglu_2()
{
    const ncnn::Mat a0 = RandomMat(10, 24);
    const ncnn::Mat b0 = RandomMat(10, 24);
    const ncnn::Mat a1 = RandomMat(7, 17, -12.f, 12.f);
    const ncnn::Mat b1 = RandomMat(7, 17, -12.f, 12.f);
    const ncnn::Mat a2 = RandomMat(33, 12, -12.f, 12.f);
    const ncnn::Mat b2 = RandomMat(33, 12, -12.f, 12.f);
    const ncnn::Mat a3 = RandomMat(17, 15);
    const ncnn::Mat b3 = RandomMat(17, 15);
    const ncnn::Mat a4 = RandomMat(256, 300);
    const ncnn::Mat b4 = RandomMat(256, 300);

    return 0
           || test_swiglu(a0, b0, 1.f, 0.f, 0.f)
           || test_swiglu(a1, b1, 1.702f, 1.f, 7.f)
           || test_swiglu(a2, b2, 1.f, 0.f, 10.f)
           || test_swiglu(a3, b3, 1.f, 1.f, 0.f)
           || test_swiglu(a3, b3, 1.f, 1.f, 0.5f)
           || test_swiglu(a4, b4, 1.f, 0.f, 0.f, TEST_LAYER_ENABLE_THREADING);
}

static int test_swiglu_3()
{
    const ncnn::Mat a0 = RandomMat(128);
    const ncnn::Mat b0 = RandomMat(128);
    const ncnn::Mat a1 = RandomMat(124, -12.f, 12.f);
    const ncnn::Mat b1 = RandomMat(124, -12.f, 12.f);
    const ncnn::Mat a2 = RandomMat(127, -12.f, 12.f);
    const ncnn::Mat b2 = RandomMat(127, -12.f, 12.f);
    const ncnn::Mat a3 = RandomMat(17);
    const ncnn::Mat b3 = RandomMat(17);

    return 0
           || test_swiglu(a0, b0, 1.f, 0.f, 0.f)
           || test_swiglu(a1, b1, 1.702f, 1.f, 7.f)
           || test_swiglu(a2, b2, 1.f, 0.f, 10.f)
           || test_swiglu(a3, b3, 1.f, 0.25f, 0.f)
           || test_swiglu(a3, b3, 1.f, 0.25f, 0.5f);
}

static int test_swiglu_4()
{
    // 1d / 2d shapes that do not divide the pack size
    const ncnn::Mat a0 = RandomMat(129);
    const ncnn::Mat b0 = RandomMat(129);
    const ncnn::Mat a1 = RandomMat(65, 7, -12.f, 12.f);
    const ncnn::Mat b1 = RandomMat(65, 7, -12.f, 12.f);
    const ncnn::Mat a2 = RandomMat(31, 33, -12.f, 12.f);
    const ncnn::Mat b2 = RandomMat(31, 33, -12.f, 12.f);

    return 0
           || test_swiglu(a0, b0, 1.f, 0.f, 0.f)
           || test_swiglu(a1, b1, 1.702f, 1.f, 7.f)
           || test_swiglu(a2, b2, 1.f, 1.f, 10.f)
           || test_swiglu(a2, b2, 1.f, 1.f, 0.5f);
}

static const float expected_standard[14] = {0.00317785081f, 0.00637735836f, -0.f, 0.f, 5.11741005f, 28.5772238f, -69.9362264f, 4.99977301f, -4.99999999f, -0.566311003f, -0.421632376f, 8800.f, -5.58011396e-42f, -150.f};
static const float expected_gptoss[14] = {2.43477678e-06f, -0.f, -0.154204234f, 0.f, 6.76636613f, 23.8554295f, -41.9997188f, 10.4999297f, 5.24996485f, -0.598446254f, -0.3023993f, 55.999625f, -3.02704717e-72f, -3.49997656f};
static const float expected_deepseek[14] = {0.00317785081f, 0.00637735836f, -0.f, 0.f, 5.11741005f, 28.5772238f, -69.9362264f, 4.99977301f, -2.49988651f, -0.566311003f, -0.421632376f, 99.9954602f, -5.58011396e-42f, -14.999319f};

static int test_swiglu_expected(const char* name, const char* mode, const ncnn::Mat& out, const float* expected, float epsilon)
{
    for (int i = 0; i < 14; i++)
    {
        if (!NearlyEqual(out[i], expected[i], epsilon))
        {
            fprintf(stderr, "swiglu oracle %s/%s failed i=%d expect %f but got %f\n", name, mode, i, expected[i], out[i]);
            return -1;
        }
    }

    return 0;
}

static int test_swiglu_oracle(const char* name, float scale, float offset, float limit, const float* expected)
{
    ncnn::ParamDict pd;
    pd.set(0, scale);
    pd.set(1, offset);
    pd.set(2, limit);

    const int size = 14;
    const float gate_data[size] = {-10.f, -7.f, -1.f, 0.f, 1.f, 3.f, 7.f, 10.f, 20.f, -0.5f, 0.25f, 88.f, -100.f, 100.f};
    const float up_data[size] = {-7.f, -1.f, 0.f, 1.f, 7.f, 10.f, -10.f, 0.5f, -0.25f, 3.f, -3.f, 100.f, 1.5f, -1.5f};

    ncnn::Mat gate(size);
    ncnn::Mat up(size);
    float* gateptr = gate;
    float* upptr = up;
    for (int i = 0; i < size; i++)
    {
        gateptr[i] = gate_data[i];
        upptr[i] = up_data[i];
    }

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;

    // the generic layer must match the reference values
    {
        ncnn::Layer* op = ncnn::create_layer_naive("SwiGLU");
        if (!op)
        {
            fprintf(stderr, "create swiglu layer failed\n");
            return -1;
        }
        op->load_param(pd);
        op->create_pipeline(opt);

        std::vector<ncnn::Mat> as(2);
        as[0] = gate;
        as[1] = up;
        std::vector<ncnn::Mat> tops(1);
        int ret = op->forward(as, tops, opt);

        op->destroy_pipeline(opt);
        delete op;

        if (ret != 0 || tops[0].empty())
        {
            fprintf(stderr, "swiglu oracle %s/naive forward failed ret=%d\n", name, ret);
            return -1;
        }
        if (test_swiglu_expected(name, "naive", tops[0], expected, 0.001f) != 0)
            return -1;
    }

    // optimized backends must match the same reference values, including the clamp edges
    {
        ncnn::Layer* op = ncnn::create_layer_cpu("SwiGLU");
        if (!op)
        {
            fprintf(stderr, "create swiglu cpu layer failed\n");
            return -1;
        }
        op->load_param(pd);
        op->create_pipeline(opt);

        std::vector<ncnn::Mat> as(2);
        as[0] = gate;
        as[1] = up;

        {
            std::vector<ncnn::Mat> tops(1);
            int ret = op->forward(as, tops, opt);
            if (ret != 0 || tops[0].empty() || test_swiglu_expected(name, "cpu", tops[0], expected, 0.001f) != 0)
            {
                fprintf(stderr, "swiglu oracle %s/cpu forward failed ret=%d\n", name, ret);
                op->destroy_pipeline(opt);
                delete op;
                return -1;
            }
        }

        if (op->support_bf16_storage)
        {
            ncnn::Option opt_bf16 = opt;
            opt_bf16.use_bf16_storage = true;
            opt_bf16.use_fp16_storage = false;
            opt_bf16.use_fp16_arithmetic = false;

            ncnn::Mat gate_bf16;
            ncnn::Mat up_bf16;
            ncnn::cast_float32_to_bfloat16(gate, gate_bf16, opt_bf16);
            ncnn::cast_float32_to_bfloat16(up, up_bf16, opt_bf16);

            as[0] = gate_bf16;
            as[1] = up_bf16;

            std::vector<ncnn::Mat> tops_bf16(1);
            int ret = op->forward(as, tops_bf16, opt_bf16);

            ncnn::Mat top;
            ncnn::cast_bfloat16_to_float32(tops_bf16[0], top, opt_bf16);

            if (ret != 0 || tops_bf16[0].empty() || test_swiglu_expected(name, "bf16", top, expected, 0.01f) != 0)
            {
                fprintf(stderr, "swiglu oracle %s/bf16 forward failed ret=%d\n", name, ret);
                op->destroy_pipeline(opt);
                delete op;
                return -1;
            }
        }

        if (op->support_fp16_storage)
        {
            ncnn::Option opt_fp16 = opt;
            opt_fp16.use_fp16_storage = true;
            opt_fp16.use_fp16_arithmetic = false;

            ncnn::Mat gate_fp16;
            ncnn::Mat up_fp16;
            ncnn::cast_float32_to_float16(gate, gate_fp16, opt_fp16);
            ncnn::cast_float32_to_float16(up, up_fp16, opt_fp16);

            as[0] = gate_fp16;
            as[1] = up_fp16;

            std::vector<ncnn::Mat> tops_fp16(1);
            int ret = op->forward(as, tops_fp16, opt_fp16);

            ncnn::Mat top;
            ncnn::cast_float16_to_float32(tops_fp16[0], top, opt_fp16);

            if (ret != 0 || tops_fp16[0].empty() || test_swiglu_expected(name, "fp16s", top, expected, 0.01f) != 0)
            {
                fprintf(stderr, "swiglu oracle %s/fp16s forward failed ret=%d\n", name, ret);
                op->destroy_pipeline(opt);
                delete op;
                return -1;
            }
        }

        op->destroy_pipeline(opt);
        delete op;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_swiglu_0()
           || test_swiglu_1()
           || test_swiglu_2()
           || test_swiglu_3()
           || test_swiglu_4()
           || test_swiglu_oracle("standard", 1.f, 0.f, 0.f, expected_standard)
           || test_swiglu_oracle("gpt-oss", 1.702f, 1.f, 7.f, expected_gptoss)
           || test_swiglu_oracle("deepseek", 1.f, 0.f, 10.f, expected_deepseek);
}
