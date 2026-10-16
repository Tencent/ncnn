#include "net.h"
#include "testutil.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif

#include <string>

static int test_batch_param_support()
{
    ncnn::Layer* layer = ncnn::create_layer("Concat");
    if (!layer)
        return -1;

    ncnn::ParamDict pd;
    pd.set(0, -233);
    int ret = layer->load_param(pd);
    delete layer;

#if NCNN_BATCH
    if (ret != 0)
#else
    if (ret == 0)
#endif
    {
        fprintf(stderr, "test_batch_param_support failed ret=%d\n", ret);
        return -1;
    }

    return 0;
}

#if NCNN_BATCH
static ncnn::Mat RandomMatBatch(int w, int n)
{
    ncnn::Mat m(w, (size_t)4u, 1, n);
    for (int b = 0; b < n; b++)
        Randomize(m.batch(b));
    return m;
}

static ncnn::Mat RandomMatBatch(int w, int h, int n)
{
    ncnn::Mat m(w, h, (size_t)4u, 1, n);
    for (int b = 0; b < n; b++)
        Randomize(m.batch(b));
    return m;
}

static ncnn::Mat RandomMatBatch(int w, int h, int c, int n)
{
    ncnn::Mat m(w, h, c, (size_t)4u, 1, n);
    for (int b = 0; b < n; b++)
        Randomize(m.batch(b));
    return m;
}

static ncnn::Mat RandomMatBatch(int w, int h, int d, int c, int n)
{
    ncnn::Mat m(w, h, d, c, (size_t)4u, 1, n);
    for (int b = 0; b < n; b++)
        Randomize(m.batch(b));
    return m;
}

static void copy_sample(const ncnn::Mat& src, ncnn::Mat& dst)
{
    for (int q = 0; q < src.c; q++)
    {
        for (int z = 0; z < src.d; z++)
        {
            for (int y = 0; y < src.h; y++)
            {
                const float* srcptr = src.channel(q).depth(z).row<const float>(y);
                float* dstptr = dst.channel(q).depth(z).row<float>(y);
                for (int x = 0; x < src.w; x++)
                    dstptr[x] = srcptr[x];
            }
        }
    }
}

static void copy_concat_sample(const std::vector<ncnn::Mat>& samples, ncnn::Mat& dst, int axis)
{
    const int dims = dst.dims;
    int offset = 0;

    for (size_t i = 0; i < samples.size(); i++)
    {
        const ncnn::Mat& src = samples[i];
        const int woffset = axis == dims - 1 ? offset : 0;
        const int hoffset = axis == dims - 2 ? offset : 0;
        const int doffset = dims == 4 && axis == 1 ? offset : 0;
        const int coffset = dims >= 3 && axis == 0 ? offset : 0;

        for (int q = 0; q < src.c; q++)
        {
            for (int z = 0; z < src.d; z++)
            {
                for (int y = 0; y < src.h; y++)
                {
                    const float* srcptr = src.channel(q).depth(z).row<const float>(y);
                    float* dstptr = dst.channel(q + coffset).depth(z + doffset).row<float>(y + hoffset) + woffset;
                    for (int x = 0; x < src.w; x++)
                        dstptr[x] = srcptr[x];
                }
            }
        }

        if (axis == dims - 1)
            offset += src.w;
        else if (axis == dims - 2)
            offset += src.h;
        else if (dims == 4 && axis == 1)
            offset += src.d;
        else if (dims >= 3 && axis == 0)
            offset += src.c;
    }
}

static ncnn::Mat create_reference(const std::vector<ncnn::Mat>& inputs, int axis)
{
    const ncnn::Mat& input0 = inputs[0];
    const int dims = input0.dims;
    int positive_axis = axis < 0 && axis != -233 ? dims + axis : axis;
    int n = 1;
    int w = input0.w;
    int h = input0.h;
    int d = input0.d;
    int c = input0.c;

    if (axis == -233)
    {
        n = 0;
        for (size_t i = 0; i < inputs.size(); i++)
            n += inputs[i].n;
    }
    else
    {
        for (size_t i = 0; i < inputs.size(); i++)
        {
            n = std::max(n, inputs[i].n);
            if (positive_axis == dims - 1)
                w += i == 0 ? 0 : inputs[i].w;
            else if (positive_axis == dims - 2)
                h += i == 0 ? 0 : inputs[i].h;
            else if (dims == 4 && positive_axis == 1)
                d += i == 0 ? 0 : inputs[i].d;
            else if (dims >= 3 && positive_axis == 0)
                c += i == 0 ? 0 : inputs[i].c;
        }
    }

    ncnn::Mat output;
    if (dims == 1)
        output.create(w, (size_t)4u, 1, n);
    if (dims == 2)
        output.create(w, h, (size_t)4u, 1, n);
    if (dims == 3)
        output.create(w, h, c, (size_t)4u, 1, n);
    if (dims == 4)
        output.create(w, h, d, c, (size_t)4u, 1, n);

    if (axis == -233)
    {
        int out_batch = 0;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            for (int b = 0; b < inputs[i].n; b++)
            {
                copy_sample(inputs[i].batch(b), output.batch(out_batch));
                out_batch++;
            }
        }
    }
    else
    {
        for (int b = 0; b < n; b++)
        {
            std::vector<ncnn::Mat> samples(inputs.size());
            for (size_t i = 0; i < inputs.size(); i++)
                samples[i] = inputs[i].n > 1 ? inputs[i].batch(b) : inputs[i].batch(0);

            copy_concat_sample(samples, output.batch(b), positive_axis);
        }
    }

    return output;
}

static std::string concat_param(const std::vector<ncnn::Mat>& inputs, int axis)
{
    std::string param = "7767517\n";
    param += std::to_string((int)inputs.size() + 1) + " " + std::to_string((int)inputs.size() + 1) + "\n";
    for (size_t i = 0; i < inputs.size(); i++)
    {
        const std::string name = "data" + std::to_string(i);
        param += "Input " + name + " 0 1 " + name + "\n";
    }

    param += "Concat concat " + std::to_string(inputs.size()) + " 1";
    for (size_t i = 0; i < inputs.size(); i++)
        param += " data" + std::to_string(i);
    param += " out 0=" + std::to_string(axis) + "\n";
    return param;
}

static int test_concat_net(const std::vector<ncnn::Mat>& inputs, int axis)
{
    const ncnn::Mat expected = create_reference(inputs, axis);
    const std::string param = concat_param(inputs, axis);

    for (int vulkan = 0; vulkan < 2; vulkan++)
    {
#if NCNN_VULKAN
        if (vulkan && ncnn::get_gpu_count() == 0)
            continue;
#else
        if (vulkan)
            continue;
#endif
        for (int packing = 0; packing < 2; packing++)
        {
            ncnn::Net net;
            net.opt.num_threads = 1;
            net.opt.use_vulkan_compute = vulkan;
            net.opt.use_packing_layout = packing;
            net.opt.use_fp16_packed = false;
            net.opt.use_fp16_storage = false;
            net.opt.use_fp16_arithmetic = false;
            if (net.load_param_mem(param.c_str()) != 0 || net.load_model((const unsigned char*)"") != 0)
                return -1;

            ncnn::Extractor ex = net.create_extractor();
            for (size_t i = 0; i < inputs.size(); i++)
            {
                if (ex.input(("data" + std::to_string(i)).c_str(), inputs[i]) != 0)
                    return -1;
            }

            ncnn::Mat output;
            int ret = ex.extract("out", output);
            if (ret != 0 || CompareMat(expected, output, 0.001f) != 0)
            {
                fprintf(stderr, "test_concat_net failed axis=%d vulkan=%d packing=%d ret=%d\n", axis, vulkan, packing, ret);
                return -1;
            }
        }
    }

    return 0;
}

static int test_concat_net_error(const std::vector<ncnn::Mat>& inputs, int axis)
{
    const std::string param = concat_param(inputs, axis);

    ncnn::Net net;
    net.opt.num_threads = 1;
    if (net.load_param_mem(param.c_str()) != 0 || net.load_model((const unsigned char*)"") != 0)
        return -1;

    ncnn::Extractor ex = net.create_extractor();
    for (size_t i = 0; i < inputs.size(); i++)
    {
        if (ex.input(("data" + std::to_string(i)).c_str(), inputs[i]) != 0)
            return -1;
    }

    ncnn::Mat output;
    if (ex.extract("out", output) == 0)
    {
        fprintf(stderr, "test_concat_net_error unexpectedly succeeded axis=%d\n", axis);
        return -1;
    }

    return 0;
}

static int test_concat_batch_axis()
{
    {
        std::vector<ncnn::Mat> inputs(3);
        inputs[0] = RandomMatBatch(13, 1);
        inputs[1] = RandomMatBatch(13, 2);
        inputs[2] = RandomMatBatch(13, 3);
        if (test_concat_net(inputs, -233) != 0)
            return -1;
    }
    {
        std::vector<ncnn::Mat> inputs(2);
        inputs[0] = RandomMatBatch(11, 7, 2);
        inputs[1] = RandomMatBatch(11, 7, 3);
        if (test_concat_net(inputs, -233) != 0)
            return -1;
    }
    {
        std::vector<ncnn::Mat> inputs(3);
        inputs[0] = RandomMatBatch(7, 5, 8, 2);
        inputs[1] = RandomMatBatch(7, 5, 8, 1);
        inputs[2] = RandomMatBatch(7, 5, 8, 3);
        if (test_concat_net(inputs, -233) != 0)
            return -1;
    }
    {
        std::vector<ncnn::Mat> inputs(2);
        inputs[0] = RandomMatBatch(5, 4, 3, 4, 2);
        inputs[1] = RandomMatBatch(5, 4, 3, 4, 2);
        if (test_concat_net(inputs, -233) != 0)
            return -1;
    }

    return 0;
}

static int test_concat_batch_inner()
{
    {
        std::vector<ncnn::Mat> inputs(2);
        inputs[0] = RandomMatBatch(9, 2);
        inputs[1] = RandomMatBatch(15, 2);
        if (test_concat_net(inputs, 0) != 0 || test_concat_net(inputs, -1) != 0)
            return -1;
    }
    {
        std::vector<ncnn::Mat> inputs(2);
        inputs[0] = RandomMatBatch(8, 5, 2);
        inputs[1] = RandomMatBatch(8, 7, 2);
        if (test_concat_net(inputs, 0) != 0 || test_concat_net(inputs, -2) != 0)
            return -1;

        inputs[0] = RandomMatBatch(5, 8, 2);
        inputs[1] = RandomMatBatch(7, 8, 2);
        if (test_concat_net(inputs, 1) != 0 || test_concat_net(inputs, -1) != 0)
            return -1;
    }
    {
        std::vector<ncnn::Mat> inputs(2);
        inputs[0] = RandomMatBatch(6, 5, 4, 2);
        inputs[1] = RandomMatBatch(6, 5, 8, 2);
        if (test_concat_net(inputs, 0) != 0 || test_concat_net(inputs, -3) != 0)
            return -1;

        inputs[0] = RandomMatBatch(6, 5, 4, 2);
        inputs[1] = RandomMatBatch(6, 7, 4, 2);
        if (test_concat_net(inputs, 1) != 0 || test_concat_net(inputs, -2) != 0)
            return -1;

        inputs[0] = RandomMatBatch(5, 6, 4, 2);
        inputs[1] = RandomMatBatch(7, 6, 4, 2);
        if (test_concat_net(inputs, 2) != 0 || test_concat_net(inputs, -1) != 0)
            return -1;
    }
    {
        std::vector<ncnn::Mat> inputs(2);
        inputs[0] = RandomMatBatch(4, 3, 2, 2, 2);
        inputs[1] = RandomMatBatch(4, 3, 2, 3, 2);
        if (test_concat_net(inputs, 0) != 0 || test_concat_net(inputs, -4) != 0)
            return -1;

        inputs[0] = RandomMatBatch(4, 4, 2, 4, 2);
        inputs[1] = RandomMatBatch(4, 4, 3, 4, 2);
        if (test_concat_net(inputs, 1) != 0 || test_concat_net(inputs, -3) != 0)
            return -1;

        inputs[0] = RandomMatBatch(4, 3, 3, 4, 2);
        inputs[1] = RandomMatBatch(4, 5, 3, 4, 2);
        if (test_concat_net(inputs, 2) != 0 || test_concat_net(inputs, -2) != 0)
            return -1;

        inputs[0] = RandomMatBatch(3, 4, 3, 4, 2);
        inputs[1] = RandomMatBatch(5, 4, 3, 4, 2);
        if (test_concat_net(inputs, 3) != 0 || test_concat_net(inputs, -1) != 0)
            return -1;
    }
    {
        std::vector<ncnn::Mat> inputs(2);
        inputs[0] = RandomMatBatch(6, 5, 4, 1);
        inputs[1] = RandomMatBatch(6, 5, 8, 3);
        if (test_concat_net(inputs, 0) != 0)
            return -1;
    }

    return 0;
}

static int test_concat_batch_errors()
{
    {
        std::vector<ncnn::Mat> inputs(2);
        inputs[0] = RandomMatBatch(10, 5, 2);
        inputs[1] = RandomMatBatch(12, 5, 2);
        if (test_concat_net_error(inputs, -233) != 0)
            return -1;
    }
    {
        std::vector<ncnn::Mat> inputs(2);
        inputs[0] = RandomMatBatch(10, 5, 2);
        inputs[1] = RandomMatBatch(12, 5, 3);
        if (test_concat_net_error(inputs, 0) != 0)
            return -1;
    }
    {
        std::vector<ncnn::Mat> inputs(2);
        inputs[0] = RandomMatBatch(10, 5, 2);
        inputs[1] = RandomMatBatch(10, 5, 3);
        if (test_concat_net_error(inputs, 0) != 0)
            return -1;
    }

    return 0;
}
#endif

int main()
{
    SRAND(7767517);

    if (test_batch_param_support() != 0)
        return -1;

#if NCNN_BATCH
    if (test_concat_batch_axis() != 0)
        return -1;
    if (test_concat_batch_inner() != 0)
        return -1;
    if (test_concat_batch_errors() != 0)
        return -1;
#endif

    return 0;
}
