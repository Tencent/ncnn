// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "cpu.h"
#include "datareader.h"
#include "gemm.h"
#include "net.h"

#if NCNN_VULKAN && NCNN_WEIGHT_QUANT
#include "command.h"

#include <memory>
#endif

#if NCNN_WEIGHT_QUANT
static void append_le(std::vector<unsigned char>& data, unsigned int value, int size)
{
    for (int i = 0; i < size; i++)
        data.push_back((unsigned char)(value >> (i * 8)));
}

static void append_float(std::vector<unsigned char>& data, float value)
{
    union
    {
        float f;
        unsigned int u;
    } v;
    v.f = value;
    append_le(data, v.u, 4);
}

static int test_gemm_int8_accumulation(int M, int N)
{
    const int K = 131071;
    char param[180];
    snprintf(param, sizeof(param), "7767517\n2 2\nInput in 0 1 a\nGemm gemm 1 1 a y 3=1 5=1 8=%d 9=%d 13=1 18=902\n", N, K);

    // The signed result fits in int32 even when the unsigned-dot compensation wraps.
    const signed char codes[] = {-128, 127, -128};
    std::vector<unsigned char> data;
    for (int j = 0; j < N; j++)
        data.insert(data.end(), K, (unsigned char)codes[j]);
    while (data.size() % 4)
        data.push_back(0);
    for (int j = 0; j < N; j++)
        append_float(data, 1.f);

    ncnn::Net net;
    net.opt.num_threads = 1;
    net.opt.use_int8_inference = true;
    net.opt.use_packing_layout = false;
    const unsigned char* ptr = data.data();
    ncnn::DataReaderFromMemory reader(ptr, data.size());
    if (net.load_param_mem(param) != 0 || net.load_model(reader) != 0 || ptr != data.data() + data.size())
        return -1;

    ncnn::Mat A(K, M);
    for (int i = 0; i < M; i++)
    {
        float* row = A.row(i);
        for (int k = 0; k < K; k++)
            row[k] = i % 2 == 0 ? 1.f : -1.f;
    }

    ncnn::Mat out;
    ncnn::Extractor ex = net.create_extractor();
    if (ex.input("a", A) != 0 || ex.extract("y", out) != 0 || out.w != N || out.h != M)
        return -1;

    for (int i = 0; i < M; i++)
    {
        const float* row = out.row(i);
        for (int j = 0; j < N; j++)
        {
            const float expected = (float)(K * codes[j] * (i % 2 == 0 ? 1 : -1));
            if (!(fabsf(row[j] - expected) <= 2.f))
            {
                fprintf(stderr, "Gemm INT8 accumulation failed M=%d N=%d i=%d j=%d expected=%g actual=%g\n", M, N, i, j, expected, row[j]);
                return -1;
            }
        }
    }

    return 0;
}

static float regular_hadamard(int i, int j, int group)
{
    float sign = 1.f;
    for (int g = group; g > 1; g /= 4)
    {
        if (i % 4 + j % 4 == 3)
            sign = -sign;
        i /= 4;
        j /= 4;
    }
    return sign / sqrtf((float)group);
}

static int test_gemm_weight_int8(int mode, int M, int K, int group, bool transA, bool transY, const ncnn::Option& opt, int output_N1M = 0, int tile = 0, int output_elempack = 0, int N = 11)
{
    if (transY && output_elempack > 1)
        N = output_elempack * 2;
    const int scale_count = mode == 900 ? K / 32 : 1;

    ncnn::Mat A(transA ? M : K, transA ? K : M);
    ncnn::Mat input(K, M);
    for (int m = 0; m < M; m++)
    {
        float* row = input.row(m);
        for (int k = 0; k < K; k++)
        {
            const float value = m == 1 ? 0.f : (float)((k * 13 + m * 5) % 61 - 30) / 32.f;
            row[k] = value;
            if (transA)
                A.row(k)[m] = value;
            else
                A.row(m)[k] = value;
        }
    }

    ncnn::Mat weights(K, N, (size_t)1u);
    std::vector<unsigned char> data;
    for (int n = 0; n < N; n++)
    {
        signed char* row = weights.row<signed char>(n);
        for (int k = 0; k < K; k++)
        {
            row[k] = (signed char)((n * 7 + k * 19) % 256 - 128);
            data.push_back((unsigned char)row[k]);
        }
    }
    while (data.size() % 4)
        data.push_back(0);

    // fp32 constant C tag followed by a per-column bias
    append_le(data, 0, 4);
    for (int n = 0; n < N; n++)
        append_float(data, (n - 5) * 0.125f);

    ncnn::Mat scales(scale_count, N);
    for (int n = 0; n < N; n++)
    {
        for (int g = 0; g < scale_count; g++)
        {
            const float scale = n == 1 && g == 0 ? 0.f : (n + 1 + g) / 1024.f;
            scales.row(n)[g] = scale;
            if (mode == 900)
                append_le(data, ncnn::float32_to_float16(scale), 2);
            else
                append_float(data, scale);
        }
    }
    while (data.size() % 4)
        data.push_back(0);

    char param[256];
    snprintf(param, sizeof(param), "7767517\n2 2\nInput in 0 1 a\nGemm gemm 1 1 a y 0=0.5 1=2.0 2=%d 3=1 5=1 6=1 8=%d 9=%d 10=4 11=%d 12=%d 14=%d 18=%d 20=%d 21=%d 22=%d 23=%d\n",
             transA, N, K, output_N1M, output_elempack, transY, mode, tile, tile, tile, group);

    ncnn::Net net;
    net.opt = opt;
    if (net.load_param_mem(param) != 0)
        return -1;

#if NCNN_VULKAN
    if (opt.use_vulkan_compute && !net.layers()[1]->support_vulkan)
        return -1;
#endif

    const unsigned char* ptr = data.data();
    ncnn::DataReaderFromMemory reader(ptr, data.size());
    if (net.load_model(reader) != 0 || ptr != data.data() + data.size())
        return -1;

    ncnn::Mat result;
    ncnn::Extractor ex = net.create_extractor();
    if (ex.input("a", A) != 0 || ex.extract("y", result, output_elempack > 1 ? 1 : 0) != 0)
        return -1;

    if (output_elempack > 1)
    {
        if (result.elempack != output_elempack)
            return -1;
        ncnn::Mat unpacked;
        ncnn::convert_packing(result, unpacked, 1, opt);
        result = unpacked;
    }

    const int outw = transY ? M : N;
    const int outh = transY ? N : M;
    ncnn::Mat reference;
    if (output_N1M)
        reference.create(outw, 1, outh);
    else
        reference.create(outw, outh);
    std::vector<float> a(K);
    const int block = scale_count == 1 ? K : 32;
    const bool dynamic = opt.use_int8_inference && mode != 901;
    for (int m = 0; m < M; m++)
    {
        const float* row = input.row(m);
        for (int k = 0; k < K; k++)
        {
            a[k] = row[k];
            if (group)
            {
                a[k] = 0.f;
                for (int j = 0; j < group; j++)
                    a[k] += row[k / group * group + j] * regular_hadamard(k % group, j, group);
            }
        }

        if (dynamic)
        {
            for (int b = 0; b < scale_count; b++)
            {
                float maxval = 0.f;
                for (int k = b * block; k < (b + 1) * block; k++)
                    maxval = std::max(maxval, fabsf(a[k]));
                if (maxval == 0.f)
                    continue;

                const float scale = maxval / 127.f;
                for (int k = b * block; k < (b + 1) * block; k++)
                    a[k] = roundf(a[k] / scale) * scale;
            }
        }

        for (int n = 0; n < N; n++)
        {
            const signed char* row_b = weights.row<const signed char>(n);
            const float* row_s = scales.row(n);
            double sum = (n - 5) * 0.25;
            for (int k = 0; k < K; k++)
                sum += a[k] * row_b[k] * row_s[k / block];

            const int i = transY ? n : m;
            const int j = transY ? m : n;
            float* row = output_N1M ? (float*)reference.channel(i) : reference.row(i);
            row[j] = (float)sum * 0.5f;
        }
    }

    bool equal = false;
    if ((dynamic || opt.use_bf16_storage) && result.dims == reference.dims && result.elemsize == 4u && result.elempack == 1 && result.w == reference.w && result.h == reference.h
        && result.c == reference.c)
    {
        double error = 0.;
        double norm = 0.;
        for (int i = 0; i < outh; i++)
        {
            const float* r = output_N1M ? (const float*)result.channel(i) : result.row(i);
            const float* ref = output_N1M ? (const float*)reference.channel(i) : reference.row(i);
            for (int j = 0; j < outw; j++)
            {
                const double diff = r[j] - ref[j];
                error += diff * diff;
                norm += (double)ref[j] * ref[j];
            }
        }
        // dynamic quantization and bf16 storage can change the rounding
        equal = error <= norm * (0.015 * 0.015);
    }
    else
    {
        equal = CompareMat(reference, result, 0.002f) == 0;
    }
    if (!equal)
    {
        fprintf(stderr, "Gemm int8 weights %d M=%d N=%d K=%d group=%d transA=%d transY=%d int8=%d vulkan=%d fp16=%d output_N1M=%d tile=%d bf16=%d\n",
                mode, M, N, K, group, transA, transY, opt.use_int8_inference, opt.use_vulkan_compute, opt.use_fp16_storage, output_N1M, tile, opt.use_bf16_storage);
        return -1;
    }

    if (!opt.use_vulkan_compute)
    {
        ncnn::Net bad;
        const unsigned char* short_ptr = data.data();
        ncnn::DataReaderFromMemory short_reader(short_ptr, data.size() - 1);
        if (bad.load_param_mem(param) != 0 || bad.load_model(short_reader) == 0)
            return -1;

        if (mode != 900)
        {
            data.resize(data.size() - 4);
            append_le(data, 0x7fc00000, 4);
            ncnn::Net nan_scale;
            const unsigned char* nan_ptr = data.data();
            ncnn::DataReaderFromMemory nan_reader(nan_ptr, data.size());
            if (nan_scale.load_param_mem(param) != 0 || nan_scale.load_model(nan_reader) == 0)
                return -1;
        }
    }

    return 0;
}

#if NCNN_VULKAN
class NonMappableGemmWeightAllocator : public ncnn::VkBlobAllocator
{
public:
    explicit NonMappableGemmWeightAllocator(const ncnn::VulkanDevice* vkdev)
        : ncnn::VkBlobAllocator(vkdev)
    {
    }

    using ncnn::VkBlobAllocator::fastMalloc;

    ncnn::VkBufferMemory* fastMalloc(size_t size)
    {
        ncnn::VkBufferMemory* mem = ncnn::VkBlobAllocator::fastMalloc(size);
        if (mem)
            mem->mapped_ptr = 0;
        return mem;
    }
};

class FailingGemmWeightStagingAllocator : public ncnn::VkWeightStagingAllocator
{
public:
    FailingGemmWeightStagingAllocator(const ncnn::VulkanDevice* vkdev, int failure)
        : ncnn::VkWeightStagingAllocator(vkdev), remaining(failure)
    {
    }

    using ncnn::VkWeightStagingAllocator::fastMalloc;

    ncnn::VkBufferMemory* fastMalloc(size_t size)
    {
        if (remaining > 0 && --remaining == 0)
            return 0;
        return ncnn::VkWeightStagingAllocator::fastMalloc(size);
    }

    int remaining;
};

static int test_gemm_weight_upload_oom(int failure)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    NonMappableGemmWeightAllocator weights(vkdev);
    FailingGemmWeightStagingAllocator weight_staging(vkdev, failure);
    ncnn::VkBlobAllocator blob(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    const int K = 64;
    const int N = 3;

    ncnn::ParamDict pd;
    pd.set(3, 1);
    pd.set(5, 1);
    pd.set(6, 1);
    pd.set(8, N);
    pd.set(9, K);
    pd.set(10, 4);
    pd.set(18, 902);
    pd.set(23, 16);

    std::vector<unsigned char> data(K * N, 7);
    append_le(data, 0, 4);
    for (int j = 0; j < N; j++)
        append_float(data, 0.25f);
    for (int j = 0; j < N; j++)
        append_float(data, 0.5f);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_int8_inference = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;
    opt.use_bf16_packed = false;
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &weight_staging;

    std::unique_ptr<ncnn::Layer> gemm(ncnn::create_layer_vulkan("Gemm"));
    gemm->vkdev = vkdev;
    const unsigned char* ptr = data.data();
    ncnn::DataReaderFromMemory reader(ptr, data.size());
    ncnn::ModelBinFromDataReader mb(reader);
    if (gemm->load_param(pd) != 0 || !gemm->support_vulkan || gemm->load_model(mb) != 0 || gemm->create_pipeline(opt) != 0)
        return -1;

    int ret = 0;
    {
        ncnn::VkTransfer cmd(vkdev);
        if (gemm->upload_model(cmd, opt) != -100 || weight_staging.remaining != 0
            || gemm->upload_model(cmd, opt) != 0 || cmd.submit_and_wait() != 0)
            ret = -1;
    }

    if (ret == 0)
    {
        opt.blob_vkallocator = &blob;
        opt.workspace_vkallocator = &blob;
        opt.staging_vkallocator = &staging;
        ncnn::Mat A(K, 3);
        A.fill(0.125f);
        ncnn::VkCompute cmd(vkdev);
        ncnn::VkMat A_gpu;
        ncnn::VkMat out_gpu;
        ncnn::Mat out;
        cmd.record_upload(A, A_gpu, opt);
        if (A_gpu.empty() || gemm->forward(A_gpu, out_gpu, cmd, opt) != 0 || out_gpu.empty())
            ret = -1;
        else
        {
            cmd.record_download(out_gpu, out, opt);
            ncnn::Mat reference(N, A.h);
            reference.fill(28.25f);
            if (cmd.submit_and_wait() != 0 || CompareMat(out, reference, 0.00001f) != 0)
                ret = -1;
        }
    }

    gemm->destroy_pipeline(opt);
    if (ret != 0)
        fprintf(stderr, "Gemm quantized weight upload retry failed staging allocation %d\n", failure);
    return ret;
}

static int test_gemm_q8_0_subnormal_scale(bool int8, bool use_subgroup_ops)
{
    const char* param = "7767517\n2 2\nInput in 0 1 a\nGemm gemm 1 1 a y 3=1 5=1 8=3 9=32 18=900\n";
    std::vector<unsigned char> data(32 * 3, 128);
    append_le(data, 0x0001, 2);
    append_le(data, 0x0002, 2);
    append_le(data, 0x8001, 2);
    append_le(data, 0, 2);

    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    net.opt.use_int8_inference = int8;
    net.opt.use_subgroup_ops = use_subgroup_ops;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_packed = false;
    net.opt.use_bf16_storage = false;

    const unsigned char* ptr = data.data();
    ncnn::DataReaderFromMemory reader(ptr, data.size());
    if (net.load_param_mem(param) != 0 || !net.layers()[1]->support_vulkan
        || net.load_model(reader) != 0 || ptr != data.data() + data.size())
        return -1;

    ncnn::Mat A(32, 1);
    A.fill(1.f);
    ncnn::Mat out;
    ncnn::Extractor ex = net.create_extractor();
    if (ex.input("a", A) != 0 || ex.extract("y", out) != 0 || out.w != 3 || out.h != 1)
        return -1;

    const float reference[3] = {-1.f / 4096.f, -1.f / 2048.f, 1.f / 4096.f};
    for (int i = 0; i < 3; i++)
    {
        if (!(fabsf(out[i] - reference[i]) <= fabsf(reference[i]) * 0.00001f))
        {
            fprintf(stderr, "Gemm Q8_0 subnormal scale failed i=%d int8=%d subgroup=%d expected=%g actual=%g\n", i, int8, use_subgroup_ops, reference[i], out[i]);
            return -1;
        }
    }

    return 0;
}

static int test_gemm_int4_packed_storage(bool bf16)
{
    ncnn::Layer* reshape = ncnn::create_layer_vulkan("Reshape");
    ncnn::Layer* rmsnorm = ncnn::create_layer_vulkan("RMSNorm");
    const bool supported = reshape && rmsnorm;
    delete reshape;
    delete rmsnorm;
    if (!supported)
        return 0;

    const char* param = "7767517\n4 4\n"
                        "Input in 0 1 a\n"
                        "Gemm gemm 1 1 a b 3=1 5=1 8=8 9=32 18=400\n"
                        "Reshape reshape 1 1 b c 0=4 1=2 2=-1\n"
                        "RMSNorm norm 1 1 c y 0=4 2=0\n";

    const signed char weights[8] = {1, -2, 3, -4, 5, -6, 7, -1};
    std::vector<unsigned char> data;
    for (int i = 0; i < 8; i++)
    {
        const unsigned char q = (unsigned char)weights[i] & 15;
        data.insert(data.end(), 16, q | (q << 4));
    }
    for (int i = 0; i < 8; i++)
        append_float(data, 4.f);

    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    net.opt.use_fp16_packed = !bf16;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_packed = bf16;
    net.opt.use_bf16_storage = false;

    const unsigned char* ptr = data.data();
    ncnn::DataReaderFromMemory reader(ptr, data.size());
    if (net.load_param_mem(param) != 0 || !net.layers()[1]->support_vulkan || !net.layers()[3]->support_vulkan
        || net.load_model(reader) != 0 || ptr != data.data() + data.size())
        return -1;

    ncnn::Mat A(32, 14);
    for (int i = 0; i < 14; i++)
    {
        float* row = A.row(i);
        for (int k = 0; k < 32; k++)
            row[k] = (i + 1) * 0.125f;
    }

    ncnn::Mat out;
    ncnn::Extractor ex = net.create_extractor();
    if (ex.input("a", A) != 0 || ex.extract("y", out) != 0 || out.w != 4 || out.h != 2 || out.c != 14)
        return -1;

    const float norms[2] = {sqrtf(30.f / 4), sqrtf(111.f / 4)};
    for (int i = 0; i < 14; i++)
    {
        const float* outptr = out.channel(i);
        for (int j = 0; j < 8; j++)
        {
            const float expected = weights[j] / norms[j / 4];
            if (!(fabsf(outptr[j] - expected) < 0.02f))
            {
                fprintf(stderr, "Gemm int4 packed storage failed bf16=%d row=%d i=%d expected=%g actual=%g\n",
                        bf16, i, j, expected, outptr[j]);
                return -1;
            }
        }
    }

    return 0;
}

static int test_gemm_int8s_broadcast(int M, int N, int K, int transA, int broadcast_type_C, int constantC, int output_N1M, int output_transpose)
{
    ncnn::Mat A(transA ? M : K, transA ? K : M);
    ncnn::Mat B(K, N, (size_t)1u);
    ncnn::Mat scales(N);
    for (int i = 0; i < A.w * A.h; i++)
        A[i] = RandomInt(-63, 64) / 64.f;

    for (int n = 0; n < N; n++)
    {
        signed char* ptr = B.row<signed char>(n);
        ptr[0] = -128;
        for (int k = 1; k < K; k++)
            ptr[k] = RandomInt(-128, 128);
        scales[n] = n % 3 == 0 ? 0.f : n % 3 == 1 ? 1.f / 128.f : 1.f / 256.f;
    }

    ncnn::Mat C;
    if (broadcast_type_C == 0)
        C = RandomMat(1);
    if (broadcast_type_C == 1)
        C = RandomMat(M);
    if (broadcast_type_C == 2)
        C = RandomMat(1, M);
    if (broadcast_type_C == 3)
        C = RandomMat(N, M);
    if (broadcast_type_C == 4)
        C = RandomMat(N);

    ncnn::ParamDict pd;
    pd.set(0, -0.5f);
    pd.set(1, 0.25f);
    pd.set(2, transA);
    pd.set(3, 1);
    pd.set(5, 1);
    pd.set(6, constantC);
    pd.set(7, M == 1 || (constantC && broadcast_type_C >= 1 && broadcast_type_C <= 3) ? M : 0);
    pd.set(8, N);
    pd.set(9, K);
    pd.set(10, constantC ? broadcast_type_C : -1);
    pd.set(11, output_N1M);
    pd.set(14, output_transpose);
    pd.set(18, 901);

    std::vector<ncnn::Mat> weights;
    weights.push_back(B);
    if (constantC && !C.empty())
        weights.push_back(C);
    weights.push_back(scales);

    std::vector<ncnn::Mat> inputs;
    inputs.push_back(A);
    if (!constantC && !C.empty())
        inputs.push_back(C);

    int ret = test_layer("Gemm", pd, weights, inputs, 1, 0.004f, TEST_LAYER_DISABLE_CPU_TESTING);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm_int8s_broadcast failed M=%d N=%d K=%d transA=%d broadcast_type_C=%d constantC=%d output_N1M=%d output_transpose=%d\n", M, N, K, transA, broadcast_type_C, constantC,
                output_N1M, output_transpose);
    }
    return ret;
}
#endif // NCNN_VULKAN
#endif // NCNN_WEIGHT_QUANT

int main()
{
#if NCNN_WEIGHT_QUANT
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_vulkan_compute = false;
    opt.use_int8_inference = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;

    ncnn::Option opt_int8 = opt;
    opt_int8.use_int8_inference = true;

    ncnn::Option opt_bf16 = opt;
    opt_bf16.use_bf16_storage = true;

    if (test_gemm_int8_accumulation(1, 1) != 0
        || test_gemm_int8_accumulation(1, 3) != 0
        || test_gemm_int8_accumulation(3, 3) != 0
        || test_gemm_weight_int8(900, 1, 64, 0, false, false, opt_int8) != 0
        || test_gemm_weight_int8(900, 1, 32, 0, false, false, opt) != 0
        || test_gemm_weight_int8(900, 1, 96, 0, false, false, opt, 0, 0, 0, 48) != 0
        || test_gemm_weight_int8(900, 5, 96, 0, true, true, opt, 1, 8, 0, 37) != 0
        || test_gemm_weight_int8(900, 5, 64, 0, true, true, opt_int8) != 0
        || test_gemm_weight_int8(900, 5, 64, 0, false, false, opt) != 0
        || test_gemm_weight_int8(901, 5, 67, 0, false, true, opt_int8) != 0
        || test_gemm_weight_int8(901, 1, 64, 16, true, false, opt_int8) != 0
        || test_gemm_weight_int8(901, 1, 67, 0, false, true, opt, 1) != 0
        || test_gemm_weight_int8(901, 17, 131, 0, true, true, opt, 1, 8) != 0
        || test_gemm_weight_int8(901, 17, 131, 0, true, true, opt_bf16, 1, 8) != 0
        || test_gemm_weight_int8(901, 1, 67, 0, false, true, opt, 1, 16, 0, 37) != 0
        || test_gemm_weight_int8(901, 17, 131, 0, true, true, opt_bf16, 1, 16, 0, 37) != 0
        || test_gemm_weight_int8(902, 1, 3, 0, false, false, opt_int8) != 0
        || test_gemm_weight_int8(902, 5, 64, 16, false, false, opt_int8) != 0
        || test_gemm_weight_int8(902, 17, 67, 0, true, true, opt_int8, 1, 8) != 0
        || test_gemm_weight_int8(902, 5, 512, 256, false, false, opt_int8) != 0
        || test_gemm_weight_int8(902, 5, 64, 16, false, false, opt, 0, 0, 0, 37) != 0)
        return 1;

#if defined(__riscv) && NCNN_RVV
    if (ncnn::cpu_support_riscv_v())
    {
        const int packn = ncnn::cpu_riscv_vlenb() / 4;
        if (test_gemm_weight_int8(902, packn, 64, 16, false, false, opt_int8, 0, 0, packn) != 0
            || test_gemm_weight_int8(901, packn, 67, 0, true, false, opt, 0, 8, packn) != 0
            || test_gemm_weight_int8(900, packn + 1, 64, 0, false, true, opt, 1, 8, packn) != 0)
            return 1;
    }
#endif

#if NCNN_VULKAN
    if (ncnn::get_gpu_count() > 0)
    {
        ncnn::Option opt_vulkan = opt;
        opt_vulkan.use_vulkan_compute = true;

        ncnn::Option opt_vulkan_int8 = opt_vulkan;
        opt_vulkan_int8.use_int8_inference = true;

        ncnn::Option opt_vulkan_bf16 = opt_vulkan;
        opt_vulkan_bf16.use_bf16_storage = true;

        ncnn::Option opt_vulkan_fp16 = opt_vulkan;
        opt_vulkan_fp16.use_fp16_packed = true;
        opt_vulkan_fp16.use_fp16_storage = true;
        opt_vulkan_fp16.use_fp16_arithmetic = true;

        ncnn::Option opt_vulkan_int8_fp16 = opt_vulkan_fp16;
        opt_vulkan_int8_fp16.use_int8_inference = true;

        ncnn::Option opt_vulkan_no_local_memory = opt_vulkan;
        opt_vulkan_no_local_memory.use_shader_local_memory = false;
        opt_vulkan_no_local_memory.use_cooperative_matrix = false;

        ncnn::Option opt_vulkan_int8_no_local_memory = opt_vulkan_no_local_memory;
        opt_vulkan_int8_no_local_memory.use_int8_inference = true;

        SRAND(7767517);
        if (test_gemm_weight_upload_oom(1) != 0
            || test_gemm_int8s_broadcast(1, 7, 5, 0, -1, 1, 0, 0) != 0
            || test_gemm_int8s_broadcast(28, 36, 91, 0, 4, 1, 0, 0) != 0
            || test_gemm_int8s_broadcast(16, 33, 17, 1, 3, 0, 0, 1) != 0
            || test_gemm_int8s_broadcast(32, 32, 127, 0, 2, 1, 1, 0) != 0
            || test_gemm_int8s_broadcast(19, 24, 64, 0, 0, 0, 1, 1) != 0
            || test_gemm_int8s_broadcast(27, 31, 32, 0, 1, 0, 0, 0) != 0
            || test_gemm_weight_upload_oom(2) != 0
            || test_gemm_weight_upload_oom(3) != 0
            || test_gemm_q8_0_subnormal_scale(false, false) != 0
            || test_gemm_q8_0_subnormal_scale(false, true) != 0
            || test_gemm_q8_0_subnormal_scale(true, false) != 0
            || test_gemm_q8_0_subnormal_scale(true, true) != 0
            || test_gemm_int4_packed_storage(false) != 0
            || test_gemm_int4_packed_storage(true) != 0
            || test_gemm_weight_int8(901, 5, 67, 0, false, true, opt_vulkan_no_local_memory) != 0
            || test_gemm_weight_int8(900, 5, 96, 0, true, true, opt_vulkan_int8_no_local_memory) != 0
            || test_gemm_weight_int8(900, 1, 64, 0, false, false, opt_vulkan_int8_no_local_memory) != 0
            || test_gemm_weight_int8(900, 5, 96, 0, true, true, opt_vulkan) != 0
            || test_gemm_weight_int8(900, 5, 96, 0, true, true, opt_vulkan_int8) != 0
            || test_gemm_weight_int8(900, 1, 64, 0, false, false, opt_vulkan_int8) != 0
            || test_gemm_weight_int8(901, 5, 67, 0, false, true, opt_vulkan_int8) != 0
            || test_gemm_weight_int8(901, 5, 67, 0, false, true, opt_vulkan_fp16) != 0
            || test_gemm_weight_int8(901, 5, 67, 0, false, true, opt_vulkan_bf16) != 0
            || test_gemm_weight_int8(902, 5, 64, 16, false, false, opt_vulkan) != 0
            || test_gemm_weight_int8(902, 5, 64, 16, false, false, opt_vulkan_int8) != 0
            || test_gemm_weight_int8(902, 5, 64, 16, false, false, opt_vulkan_int8_fp16) != 0
            || test_gemm_weight_int8(902, 1, 67, 0, true, true, opt_vulkan_int8) != 0)
            return 1;
    }
#endif // NCNN_VULKAN
#endif // NCNN_WEIGHT_QUANT
    return 0;
}
