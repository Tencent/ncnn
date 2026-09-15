// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "datareader.h"
#include "layer_type.h"

static int test_innerproduct(const ncnn::Mat& a, int outch, int bias)
{
    ncnn::ParamDict pd;
    pd.set(0, outch); // num_output
    pd.set(1, bias);  // bias_term
    pd.set(2, outch * a.w * a.h * a.d * a.c);

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * a.w * a.h * a.d * a.c);
    if (bias)
        weights[1] = RandomMat(outch);

    int ret = test_layer("InnerProduct", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_innerproduct failed a.dims=%d a=(%d %d %d %d) outch=%d bias=%d act=%d actparams=[%f,%f]\n", a.dims, a.w, a.h, a.d, a.c, outch, bias, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_innerproduct_0()
{
    return 0
           || test_innerproduct(RandomMat(1, 3, 1), 1, 1)
           || test_innerproduct(RandomMat(3, 2, 2), 2, 0)
           || test_innerproduct(RandomMat(9, 3, 8), 7, 1)
           || test_innerproduct(RandomMat(2, 2, 8), 8, 0)
           || test_innerproduct(RandomMat(4, 3, 15), 8, 1)
           || test_innerproduct(RandomMat(6, 2, 16), 16, 0)
           || test_innerproduct(RandomMat(6, 2, 16), 7, 1)
           || test_innerproduct(RandomMat(6, 2, 5), 16, 1);
}

static int test_innerproduct_1()
{
    return 0
           || test_innerproduct(RandomMat(1, 1), 1, 1)
           || test_innerproduct(RandomMat(3, 2), 2, 0)
           || test_innerproduct(RandomMat(9, 8), 7, 1)
           || test_innerproduct(RandomMat(2, 8), 8, 0)
           || test_innerproduct(RandomMat(4, 15), 8, 1)
           || test_innerproduct(RandomMat(6, 16), 16, 0)
           || test_innerproduct(RandomMat(6, 16), 7, 1)
           || test_innerproduct(RandomMat(6, 5), 16, 1);
}

static int test_innerproduct_2()
{
    return 0
           || test_innerproduct(RandomMat(1), 1, 1)
           || test_innerproduct(RandomMat(2), 2, 0)
           || test_innerproduct(RandomMat(8), 7, 1)
           || test_innerproduct(RandomMat(8), 8, 0)
           || test_innerproduct(RandomMat(15), 8, 1)
           || test_innerproduct(RandomMat(15), 15, 1)
           || test_innerproduct(RandomMat(16), 16, 0)
           || test_innerproduct(RandomMat(16), 7, 1)
           || test_innerproduct(RandomMat(5), 16, 0)
           || test_innerproduct(RandomMat(32), 16, 1)
           || test_innerproduct(RandomMat(12), 16, 0)
           || test_innerproduct(RandomMat(16), 12, 1)
           || test_innerproduct(RandomMat(24), 32, 1);
}

static int test_innerproduct_3()
{
    return 0
           || test_innerproduct(RandomMat(2, 2, 3, 1), 8, 1)
           || test_innerproduct(RandomMat(5, 3, 2, 3), 7, 0)
           || test_innerproduct(RandomMat(3, 2, 2, 5), 8, 1)
           || test_innerproduct(RandomMat(4, 3, 3, 4), 15, 1)
           || test_innerproduct(RandomMat(2, 2, 3, 16), 16, 0);
}

#if NCNN_INT8
static int test_innerproduct_int8(const ncnn::Mat& a, int outch, int bias, bool input_int8 = false, bool weight_int8 = false)
{
    ncnn::ParamDict pd;
    pd.set(0, outch); // num_output
    pd.set(1, bias);  // bias_term
    pd.set(2, outch * a.w * a.h * a.d * a.c);
    pd.set(8, 1); // int8_scale_term

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 4 : 3);
    const int k = a.w * a.h * a.d * a.c;
    weights[0] = weight_int8 ? RandomS8Mat(outch * k) : RandomMat(outch * k);
    ncnn::Mat weight_scales = weight_int8 ? RandomMat(outch, 10.f, 20.f) : scales_mat(weights[0], outch, k, k);
    if (!weight_int8)
    {
        for (int q = 0; q < outch; q++)
        {
            weight_scales[q] = std::min(weight_scales[q], 127.f);
        }
    }
    ncnn::Mat input_scales = scales_mat(a, 1, k, k);
    input_scales[0] = std::min(input_scales[0], 127.f);

    ncnn::Mat a_int8 = a;
    if (input_int8)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = false;
        ncnn::quantize_to_int8(a, a_int8, input_scales, opt);
    }

    if (bias)
    {
        weights[1] = RandomMat(outch);
        weights[2] = weight_scales;
        weights[3] = input_scales;
    }
    else
    {
        weights[1] = weight_scales;
        weights[2] = input_scales;
    }

    int flag = input_int8 ? TEST_LAYER_DISABLE_AUTO_INPUT_CASTING : 0;
    int ret = 0;
    if (input_int8)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_packed = false;
        opt.use_bf16_storage = false;

        ret = test_layer_opt("InnerProduct", pd, weights, opt, a_int8, 0.001f, flag);
    }
    else
    {
        ret = test_layer("InnerProduct", pd, weights, a_int8, 0.001f, flag);
    }
    if (ret != 0)
    {
        fprintf(stderr, "test_innerproduct_int8 failed a.dims=%d a=(%d %d %d %d) outch=%d bias=%d input_int8=%d weight_int8=%d act=%d actparams=[%f,%f]\n", a.dims, a.w, a.h, a.d, a.c, outch, bias, input_int8, weight_int8, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_innerproduct_4()
{
    return 0
           || test_innerproduct_int8(RandomMat(1, 3, 1), 1, 1)
           || test_innerproduct_int8(RandomMat(3, 2, 2), 2, 1)
           || test_innerproduct_int8(RandomMat(5, 3, 3), 3, 1)
           || test_innerproduct_int8(RandomMat(7, 2, 3), 12, 1)
           || test_innerproduct_int8(RandomMat(9, 3, 4), 4, 1)
           || test_innerproduct_int8(RandomMat(2, 2, 7), 7, 1)
           || test_innerproduct_int8(RandomMat(4, 3, 8), 3, 1)
           || test_innerproduct_int8(RandomMat(6, 2, 8), 8, 1)
           || test_innerproduct_int8(RandomMat(8, 3, 15), 15, 1)
           || test_innerproduct_int8(RandomMat(7, 2, 16), 4, 1)
           || test_innerproduct_int8(RandomMat(6, 3, 16), 16, 1)
           || test_innerproduct_int8(RandomMat(16), 16, 1, true)
           || test_innerproduct_int8(RandomMat(32), 16, 1, false, true)
           || test_innerproduct_int8(RandomMat(16), 12, 1, true, true)
           || test_innerproduct_int8(RandomMat(2, 2, 1), 7, 1, true)
           || test_innerproduct_int8(RandomMat(2, 2, 2), 7, 1, true)
           || test_innerproduct_int8(RandomMat(2, 2, 3), 7, 1, true)
           || test_innerproduct_int8(RandomMat(2, 2, 4), 8, 1, true);
}
#endif // NCNN_INT8

static int test_innerproduct_gemm(const ncnn::Mat& a, int outch, int bias)
{
    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, bias);
    pd.set(2, outch * a.w);

    int activation_type = RAND() % 7;
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * a.w);
    if (bias)
        weights[1] = RandomMat(outch);

    int ret = test_layer("InnerProduct", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_innerproduct_gemm failed a.dims=%d a=(%d %d %d) outch=%d bias=%d act=%d actparams=[%f,%f]\n", a.dims, a.w, a.h, a.c, outch, bias, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_innerproduct_5()
{
    return 0
           || test_innerproduct_gemm(RandomMat(1, 1), 1, 1)
           || test_innerproduct_gemm(RandomMat(48, 1), 11, 1)
           || test_innerproduct_gemm(RandomMat(1, 5), 1, 1)
           || test_innerproduct_gemm(RandomMat(3, 2), 2, 0)
           || test_innerproduct_gemm(RandomMat(9, 8), 7, 1)
           || test_innerproduct_gemm(RandomMat(2, 8), 8, 0)
           || test_innerproduct_gemm(RandomMat(13, 20), 8, 1)
           || test_innerproduct_gemm(RandomMat(16, 20), 16, 0)
           || test_innerproduct_gemm(RandomMat(11, 24), 8, 0)
           || test_innerproduct_gemm(RandomMat(13, 24), 12, 1)
           || test_innerproduct_gemm(RandomMat(15, 20), 20, 1)
           || test_innerproduct_gemm(RandomMat(16, 20), 11, 1)
           || test_innerproduct_gemm(RandomMat(19, 16), 16, 1)
           || test_innerproduct_gemm(RandomMat(15, 15), 15, 1)
           || test_innerproduct_gemm(RandomMat(14, 15), 8, 1)
           || test_innerproduct_gemm(RandomMat(17, 15), 12, 1)
           || test_innerproduct_gemm(RandomMat(12, 16), 7, 1)
           || test_innerproduct_gemm(RandomMat(11, 32), 32, 1)
           || test_innerproduct_gemm(RandomMat(12, 32), 24, 1)
           || test_innerproduct_gemm(RandomMat(13, 32), 12, 1)
           || test_innerproduct_gemm(RandomMat(14, 32), 14, 1)
           || test_innerproduct_gemm(RandomMat(15, 32), 32, 1)
           || test_innerproduct_gemm(RandomMat(16, 24), 32, 1)
           || test_innerproduct_gemm(RandomMat(17, 20), 32, 1)
           || test_innerproduct_gemm(RandomMat(18, 14), 32, 1);
}

#if NCNN_INT8
static int test_innerproduct_gemm_int8(const ncnn::Mat& a, int outch, int bias, bool input_int8 = false, bool weight_int8 = false)
{
    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, bias);
    pd.set(2, outch * a.w);
    pd.set(8, 1); // int8_scale_term

    std::vector<ncnn::Mat> weights(bias ? 4 : 3);
    const int k = a.w;
    weights[0] = weight_int8 ? RandomS8Mat(outch * k) : RandomMat(outch * k);
    ncnn::Mat weight_scales = weight_int8 ? RandomMat(outch, 10.f, 20.f) : scales_mat(weights[0], outch, k, k);
    if (!weight_int8)
    {
        for (int q = 0; q < outch; q++)
        {
            weight_scales[q] = std::min(weight_scales[q], 127.f);
        }
    }
    ncnn::Mat input_scales = scales_mat(a, 1, k, k);
    input_scales[0] = std::min(input_scales[0], 127.f);

    ncnn::Mat a_int8 = a;
    if (input_int8)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = false;
        ncnn::quantize_to_int8(a, a_int8, input_scales, opt);
    }

    if (bias)
    {
        weights[1] = RandomMat(outch);
        weights[2] = weight_scales;
        weights[3] = input_scales;
    }
    else
    {
        weights[1] = weight_scales;
        weights[2] = input_scales;
    }

    int flag = input_int8 ? TEST_LAYER_DISABLE_AUTO_INPUT_CASTING : 0;
    int ret = 0;
    if (input_int8)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_packed = false;
        opt.use_bf16_storage = false;

        ret = test_layer_opt("InnerProduct", pd, weights, opt, a_int8, 0.001f, flag);
    }
    else
    {
        ret = test_layer("InnerProduct", pd, weights, a_int8, 0.001f, flag);
    }
    if (ret != 0)
    {
        fprintf(stderr, "test_innerproduct_gemm_int8 failed a.dims=%d a=(%d %d %d) outch=%d bias=%d input_int8=%d weight_int8=%d\n", a.dims, a.w, a.h, a.c, outch, bias, input_int8, weight_int8);
    }

    return ret;
}

static int test_innerproduct_6()
{
    return 0
           || test_innerproduct_gemm_int8(RandomMat(1, 5), 1, 1)
           || test_innerproduct_gemm_int8(RandomMat(3, 2), 2, 0)
           || test_innerproduct_gemm_int8(RandomMat(9, 8), 7, 1)
           || test_innerproduct_gemm_int8(RandomMat(2, 8), 8, 0)
           || test_innerproduct_gemm_int8(RandomMat(13, 12), 8, 1)
           || test_innerproduct_gemm_int8(RandomMat(16, 12), 16, 0)
           || test_innerproduct_gemm_int8(RandomMat(4, 15), 8, 1)
           || test_innerproduct_gemm_int8(RandomMat(6, 16), 16, 0)
           || test_innerproduct_gemm_int8(RandomMat(12, 16), 7, 1)
           || test_innerproduct_gemm_int8(RandomMat(11, 16), 8, 1, false, true)
           || test_innerproduct_gemm_int8(RandomMat(13, 15), 7, 1, true)
           || test_innerproduct_gemm_int8(RandomMat(12, 16), 7, 1, true)
           || test_innerproduct_gemm_int8(RandomMat(12, 16), 7, 1, true, true);
}

static int test_innerproduct_7()
{
    return 0
           || test_innerproduct_int8(RandomMat(2, 2, 2, 1), 5, 1)
           || test_innerproduct_int8(RandomMat(3, 2, 2, 3), 7, 1)
           || test_innerproduct_int8(RandomMat(2, 2, 3, 4), 8, 1);
}
#endif // NCNN_INT8

static int test_innerproduct_load_param_case(const ncnn::ParamDict& pd, bool valid)
{
    ncnn::Layer* layer = ncnn::create_layer_naive(ncnn::LayerType::InnerProduct);
    if (!layer)
        return -1;

    int ret = layer->load_param(pd);
    delete layer;

    if (ret != (valid ? 0 : -1))
    {
        const int num_output = pd.get(0, 0);
        const int weight_data_size = pd.get(2, 0);
        const int int8_scale_term = pd.get(8, 0);
        const int activation_type = pd.get(9, 0);

        fprintf(stderr, "test_innerproduct_load_param failed ret=%d expected=%d num_output=%d weight_data_size=%d int8_scale_term=%d activation_type=%d\n", ret, valid ? 0 : -1, num_output, weight_data_size, int8_scale_term, activation_type);

        const ncnn::Mat activation_params = pd.get(10, ncnn::Mat());
        fprintf(stderr, "activation_params type=%d dims=%d w=%d elemsize=%zu elempack=%d\n", pd.type(10), activation_params.dims, activation_params.w, activation_params.elemsize, activation_params.elempack);
        return -1;
    }

    return 0;
}

static int test_innerproduct_load_param_activation(const ncnn::ParamDict& base, int activation_type, const ncnn::Mat& activation_params, bool valid)
{
    ncnn::ParamDict pd = base;
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    return test_innerproduct_load_param_case(pd, valid);
}

#if NCNN_STRING
class InnerProductParamDict : public ncnn::ParamDict
{
public:
    using ncnn::ParamDict::load_param;
};

static int test_innerproduct_param_text(float value, float expected)
{
    InnerProductParamDict pd;
    const unsigned char* text = (const unsigned char*)"0=8 2=64 9=3 -23310=2,-1,2";
    ncnn::DataReaderFromMemory reader(text);
    if (pd.load_param(reader) != 0)
        return -1;

    if (test_innerproduct_load_param_case(pd, true) != 0)
        return -1;

    pd.set(0, 1);
    pd.set(2, 1);
    if (test_innerproduct_load_param_case(pd, true) != 0)
        return -1;

    std::vector<ncnn::Mat> weights(1);
    weights[0].create(1);
    weights[0][0] = 1.f;

    ncnn::Mat a(1);
    a[0] = value;
    ncnn::Mat reference(1);
    reference[0] = expected;
    ncnn::Mat b;
    int ret = test_layer_naive(ncnn::LayerType::InnerProduct, pd, weights, a, b, 0);
    if (ret == 0)
        ret = CompareMat(reference, b, 0.f);
    if (ret != 0)
    {
        fprintf(stderr, "test_innerproduct_param_text failed value=%f expected=%f ret=%d\n", value, expected, ret);
        return ret;
    }

    const ncnn::Mat original = pd.get(10, ncnn::Mat());
    const int* p = original;
    if (p[0] != -1 || p[1] != 2)
    {
        fprintf(stderr, "test_innerproduct_param_text modified params=[%d,%d]\n", p[0], p[1]);
        return -1;
    }

    return 0;
}
#endif

static int test_innerproduct_param_text()
{
#if NCNN_STRING
    return 0
           || test_innerproduct_param_text(3.f, 2.f)
           || test_innerproduct_param_text(-3.f, -1.f);
#else
    return 0;
#endif
}

static int test_innerproduct_load_param()
{
    ncnn::ParamDict base;
    base.set(0, 8);
    base.set(2, 64);
    if (test_innerproduct_load_param_case(base, true) != 0)
        return -1;

    ncnn::Mat params(2);
    params[0] = 0.1f;
    params[1] = 0.5f;

    int ret = 0
              || test_innerproduct_load_param_activation(base, 0, ncnn::Mat(), true)
              || test_innerproduct_load_param_activation(base, 0, params, true)
              || test_innerproduct_load_param_activation(base, 0, params.range(0, 1), true)
              || test_innerproduct_load_param_activation(base, 1, ncnn::Mat(), true)
              || test_innerproduct_load_param_activation(base, 1, params, true)
              || test_innerproduct_load_param_activation(base, 1, params.range(0, 1), true)
              || test_innerproduct_load_param_activation(base, 2, ncnn::Mat(), false)
              || test_innerproduct_load_param_activation(base, 2, params, true)
              || test_innerproduct_load_param_activation(base, 2, params.range(0, 1), true)
              || test_innerproduct_load_param_activation(base, 3, ncnn::Mat(), false)
              || test_innerproduct_load_param_activation(base, 3, params, true)
              || test_innerproduct_load_param_activation(base, 3, params.range(0, 1), false)
              || test_innerproduct_load_param_activation(base, 4, ncnn::Mat(), true)
              || test_innerproduct_load_param_activation(base, 4, params, true)
              || test_innerproduct_load_param_activation(base, 4, params.range(0, 1), true)
              || test_innerproduct_load_param_activation(base, 5, ncnn::Mat(), true)
              || test_innerproduct_load_param_activation(base, 5, params, true)
              || test_innerproduct_load_param_activation(base, 5, params.range(0, 1), true)
              || test_innerproduct_load_param_activation(base, 6, ncnn::Mat(), false)
              || test_innerproduct_load_param_activation(base, 6, params, true)
              || test_innerproduct_load_param_activation(base, 6, params.range(0, 1), false)
              || test_innerproduct_load_param_activation(base, 7, ncnn::Mat(), false);
    if (ret != 0)
        return ret;

    const ncnn::Mat bad[] = {ncnn::Mat(2, (size_t)1u), ncnn::Mat(2, (size_t)2u), ncnn::Mat(2, 2), ncnn::Mat(2, (size_t)16u, 4)};
    for (int i = 0; i < 4; i++)
    {
        ncnn::ParamDict pd = base;
        pd.set(10, bad[i]);
        if (test_innerproduct_load_param_case(pd, false) != 0)
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

#if NCNN_INT8
    return 0
           || test_innerproduct_0()
           || test_innerproduct_1()
           || test_innerproduct_2()
           || test_innerproduct_3()
           || test_innerproduct_4()
           || test_innerproduct_5()
           || test_innerproduct_6()
           || test_innerproduct_7()
           || test_innerproduct_load_param()
           || test_innerproduct_param_text();
#else
    return 0
           || test_innerproduct_0()
           || test_innerproduct_1()
           || test_innerproduct_2()
           || test_innerproduct_3()
           || test_innerproduct_5()
           || test_innerproduct_load_param()
           || test_innerproduct_param_text();
#endif
}
