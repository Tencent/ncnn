// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include "layer/scale.h"

#include <limits.h>

#if NCNN_VALIDATION
class ScaleValidationProbe : public ncnn::Scale
{
public:
    using ncnn::Scale::validate_inputs;
};

static int test_scale_validation()
{
    ScaleValidationProbe layer;
    ncnn::Option opt;
    layer.bias_term = 0;

    // Exercise the validator itself; invalid objects never reach a compute kernel.
    std::vector<ncnn::Mat> inputs;
    if (layer.validate_inputs(inputs, opt) != -1)
        return -1;
    inputs.resize(1);
    if (layer.validate_inputs(inputs, opt) != -1)
        return -1;
    inputs.resize(3);
    if (layer.validate_inputs(inputs, opt) != -1)
        return -1;
    inputs.resize(2);
    if (layer.validate_inputs(inputs, opt) != -1)
        return -1;

    const int packs[] = {1, 4, 8, 16};
    for (int dims = 1; dims <= 4; dims++)
    {
        for (int p = 0; p < 4; p++)
        {
            const int pack = packs[p];
            ncnn::Mat a;
            if (dims == 1) a.create(3, (size_t)4 * pack, pack);
            if (dims == 2) a.create(3, 2, (size_t)4 * pack, pack);
            if (dims == 3) a.create(3, 2, 2, (size_t)4 * pack, pack);
            if (dims == 4) a.create(3, 2, 2, 2, (size_t)4 * pack, pack);
            const int required = (dims == 1 ? 3 : 2) * pack;
            inputs[0] = a;
            inputs[1] = ncnn::Mat(required);
            layer.bias_term = 0;
            if (layer.validate_inputs(inputs, opt) != 0)
                return -1;

            inputs[1] = ncnn::Mat(required + 1);
            if (layer.validate_inputs(inputs, opt) != 0)
                return -1;
            inputs[1] = ncnn::Mat(required - 1);
            if (layer.validate_inputs(inputs, opt) != -1)
                return -1;
            inputs[1] = ncnn::Mat();
            if (layer.validate_inputs(inputs, opt) != -1)
                return -1;
            inputs[1] = ncnn::Mat(required / pack, (size_t)4 * pack, pack);
            if (layer.validate_inputs(inputs, opt) != 0)
                return -1;

            layer.bias_term = 1;
            layer.bias_data = ncnn::Mat();
            if (layer.validate_inputs(inputs, opt) != -1)
                return -1;
            layer.bias_data = ncnn::Mat(required - 1);
            if (layer.validate_inputs(inputs, opt) != -1)
                return -1;
            layer.bias_data = ncnn::Mat(required);
            if (layer.validate_inputs(inputs, opt) != 0)
                return -1;
            layer.bias_data = ncnn::Mat(required + 1);
            if (layer.validate_inputs(inputs, opt) != 0)
                return -1;
        }
    }

    layer.bias_term = 1;
    inputs[0] = ncnn::Mat(3);
    inputs[1] = ncnn::Mat(3);
    layer.bias_data = ncnn::Mat(3);
    // Independently reject invalid shape, packing and scalar storage for each object.
    for (int object = 0; object < 3; object++)
    {
        ncnn::Mat& m = object == 2 ? layer.bias_data : inputs[object];
        const ncnn::Mat valid = m;
        m.dims = 0;
        if (layer.validate_inputs(inputs, opt) != -1) return -1;
        m = valid;
        m.dims = 5;
        if (layer.validate_inputs(inputs, opt) != -1) return -1;
        m = valid;
        m.w = 0;
        if (layer.validate_inputs(inputs, opt) != -1) return -1;
        m = valid;
        m.elempack = 0;
        if (layer.validate_inputs(inputs, opt) != -1) return -1;
        m = valid;
        m.elempack = INT_MAX;
        if (layer.validate_inputs(inputs, opt) != -1) return -1;
        m = valid;
        m.elemsize = 1;
        if (layer.validate_inputs(inputs, opt) != -1) return -1;
        m = valid;
        m.w = INT_MAX;
        m.elempack = 4;
        m.elemsize = 16;
        if (layer.validate_inputs(inputs, opt) != -1) return -1;
        m = valid;
    }

    // A two-dimensional coefficient object is not the contiguous vector contract.
    inputs[1] = ncnn::Mat(3, 2);
    if (layer.validate_inputs(inputs, opt) != -1) return -1;
    inputs[1] = ncnn::Mat(3);
    layer.bias_data = ncnn::Mat(3, 2);
    if (layer.validate_inputs(inputs, opt) != -1) return -1;
    layer.bias_data = ncnn::Mat(3);

    // Check the remaining input dimensions without allocating large buffers.
    const ncnn::Mat valid = inputs[0];
    inputs[0].h = 0;
    if (layer.validate_inputs(inputs, opt) != -1) return -1;
    inputs[0] = valid;
    inputs[0].d = 0;
    if (layer.validate_inputs(inputs, opt) != -1) return -1;
    inputs[0] = valid;
    inputs[0].c = 0;
    if (layer.validate_inputs(inputs, opt) != -1) return -1;
    inputs[0] = valid;
    inputs[0].h = INT_MAX;
    if (layer.validate_inputs(inputs, opt) != -1) return -1;
    inputs[0] = valid;
    inputs[0].d = INT_MAX;
    if (layer.validate_inputs(inputs, opt) != -1) return -1;
    inputs[0] = valid;
    inputs[0].dims = 3;
    inputs[0].c = INT_MAX;
    inputs[0].elempack = 4;
    inputs[0].elemsize = 16;
    if (layer.validate_inputs(inputs, opt) != -1) return -1;

    // 16-bit coefficients are accepted only with the matching input storage path.
    inputs[0] = ncnn::Mat(3);
    inputs[1] = ncnn::Mat(3, (size_t)2);
    if (layer.validate_inputs(inputs, opt) != -1) return -1;
    inputs[0] = ncnn::Mat(3, (size_t)2);
    if (layer.validate_inputs(inputs, opt) != -1) return -1;
    layer.support_bf16_storage = true;
    opt.use_bf16_storage = true;
    if (layer.validate_inputs(inputs, opt) != 0) return -1;
    inputs[1] = ncnn::Mat(3);
    if (layer.validate_inputs(inputs, opt) != 0) return -1;
    layer.bias_data = ncnn::Mat(3, (size_t)2);
    if (layer.validate_inputs(inputs, opt) != -1) return -1;

    return 0;
}
#endif // NCNN_VALIDATION

static int test_scale(const ncnn::Mat& a, int bias, int flag = 0)
{
    int scale_data_size;
    if (a.dims == 1) scale_data_size = a.w;
    if (a.dims == 2) scale_data_size = a.h;
    if (a.dims == 3) scale_data_size = a.c;
    if (a.dims == 4) scale_data_size = a.c;

    ncnn::ParamDict pd;
    pd.set(0, scale_data_size);
    pd.set(1, bias);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(scale_data_size);
    if (bias)
        weights[1] = RandomMat(scale_data_size);

    int ret = test_layer("Scale", pd, weights, a, 0.001, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_scale failed a.dims=%d a=(%d %d %d %d) bias=%d\n", a.dims, a.w, a.h, a.d, a.c, bias);
    }

    return ret;
}

static int test_scale_attention(const ncnn::Mat& a, int flag = 0)
{
    int scale_data_size;
    if (a.dims == 1) scale_data_size = a.w;
    if (a.dims == 2) scale_data_size = a.h;
    if (a.dims == 3) scale_data_size = a.c;
    if (a.dims == 4) scale_data_size = a.c;

    ncnn::ParamDict pd;
    pd.set(0, -233);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> ab(2);
    ab[0] = a;
    ab[1] = RandomMat(scale_data_size);

    int ret = test_layer("Scale", pd, weights, ab, 2, 0.001, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_scale_attention failed a.dims=%d a=(%d %d %d %d)\n", a.dims, a.w, a.h, a.d, a.c);
    }

    return ret;
}

// cpu pack8/pack16 cases reuse the Vulkan pack4 path covered by the pack4 cases
// keep the 1d sizes for dispatch boundary coverage
static int test_scale_0()
{
    return 0
           || test_scale(RandomMat(5, 3, 48), 0, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_scale(RandomMat(5, 3, 48), 1, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_scale(RandomMat(5, 7, 24), 0, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_scale(RandomMat(5, 7, 24), 1, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_scale(RandomMat(7, 9, 12), 0)
           || test_scale(RandomMat(7, 9, 12), 1)
           || test_scale(RandomMat(3, 5, 13), 0)
           || test_scale(RandomMat(3, 5, 13), 1);
}

static int test_scale_1()
{
    return 0
           || test_scale(RandomMat(13, 48), 0, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_scale(RandomMat(13, 48), 1, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_scale(RandomMat(15, 24), 0, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_scale(RandomMat(15, 24), 1, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_scale(RandomMat(17, 12), 0)
           || test_scale(RandomMat(17, 12), 1)
           || test_scale(RandomMat(19, 15), 0)
           || test_scale(RandomMat(19, 15), 1);
}

static int test_scale_2()
{
    return 0
           || test_scale(RandomMat(128), 0)
           || test_scale(RandomMat(128), 1)
           || test_scale(RandomMat(124), 0)
           || test_scale(RandomMat(124), 1)
           || test_scale(RandomMat(127), 0)
           || test_scale(RandomMat(127), 1);
}

static int test_scale_3()
{
    return 0
           || test_scale_attention(RandomMat(5, 6, 48), TEST_LAYER_DISABLE_GPU_TESTING)
           || test_scale_attention(RandomMat(5, 7, 24), TEST_LAYER_DISABLE_GPU_TESTING)
           || test_scale_attention(RandomMat(7, 9, 12))
           || test_scale_attention(RandomMat(3, 5, 13));
}

static int test_scale_4()
{
    return 0
           || test_scale_attention(RandomMat(25, 48), TEST_LAYER_DISABLE_GPU_TESTING)
           || test_scale_attention(RandomMat(15, 24), TEST_LAYER_DISABLE_GPU_TESTING)
           || test_scale_attention(RandomMat(17, 12))
           || test_scale_attention(RandomMat(19, 15));
}

static int test_scale_5()
{
    return 0
           || test_scale_attention(RandomMat(128))
           || test_scale_attention(RandomMat(124))
           || test_scale_attention(RandomMat(127));
}

static int test_scale_6()
{
    return 0
           || test_scale(RandomMat(5, 3, 2, 24), 0)
           || test_scale(RandomMat(5, 3, 2, 24), 1)
           || test_scale(RandomMat(3, 5, 3, 13), 0)
           || test_scale(RandomMat(3, 5, 3, 13), 1)
           || test_scale_attention(RandomMat(5, 3, 2, 24))
           || test_scale_attention(RandomMat(3, 5, 3, 13));
}

int main()
{
    SRAND(7767517);

    return 0
#if NCNN_VALIDATION
           || test_scale_validation()
#endif
           || test_scale_6()
           || test_scale_0()
           || test_scale_1()
           || test_scale_2()
           || test_scale_3()
           || test_scale_4()
           || test_scale_5();
}
