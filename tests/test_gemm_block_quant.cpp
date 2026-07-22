// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#if NCNN_WEIGHT_QUANT
static int weight_block_quantize_term(int bits, int block_size, int has_input_scale = 0)
{
    const int block_size_code = block_size == 32 ? 0 : block_size == 64 ? 1 : 2;
    return bits * 100 + (has_input_scale ? 10 : 0) + block_size_code;
}

static ncnn::Mat make_input_scales(int K)
{
    ncnn::Mat input_scales(K);
    float* ptr = input_scales;
    const float scale_table[5] = {0.5f, 1.f, 2.f, 0.25f, 4.f};
    for (int k = 0; k < K; k++)
        ptr[k] = scale_table[k % 5];

    return input_scales;
}

static void RandomizeA(ncnn::Mat& A, int transA, int block_size, const ncnn::Mat& input_scales)
{
    int M = A.h;
    if (transA)
        M = A.w;
    if (A.dims == 3)
        M = A.c;

    const int K = transA ? A.h : A.w;
    const float* input_scale_ptr = input_scales;

    for (int i = 0; i < M; i++)
    {
        float* ptr = 0;
        if (!transA)
            ptr = A.dims == 3 ? A.channel(i) : A.row(i);

        for (int k = 0; k < K; k++)
        {
            int q = RandomInt(-120, 121);
            if (k % block_size == 0)
                q = (i + k / block_size) % 2 == 0 ? 127 : -127;

            float v = q / 64.f;
            if (input_scale_ptr)
                v /= input_scale_ptr[k];

            if (transA)
                A.row(k)[i] = v;
            else
                ptr[k] = v;
        }
    }
}

static void RandomizeB(ncnn::Mat& B, int block_size, const ncnn::Mat& input_scales)
{
    const int K = B.w;
    const int N = B.h;
    const float* input_scale_ptr = input_scales;

    for (int n = 0; n < N; n++)
    {
        float* ptr = B.row(n);
        for (int k = 0; k < K; k++)
        {
            int q = RandomInt(-120, 121);
            if (k % block_size == 0)
                q = (n + k / block_size) % 2 == 0 ? 127 : -127;

            float v = q / 64.f;
            if (input_scale_ptr)
                v *= input_scale_ptr[k];

            ptr[k] = v;
        }
    }
}

static void quantize_weight(const ncnn::Mat& weight_data, int bits, int block_size, ncnn::Mat& weight_data_quantized, ncnn::Mat& weight_data_quantize_scales, ncnn::Mat& weight_data_dequantized)
{
    const int K = weight_data.w;
    const int N = weight_data.h;
    const int block_count = (K + block_size - 1) / block_size;

    weight_data_quantized.create((K * bits + 7) / 8, N, (size_t)1u);
    weight_data_quantize_scales.create(block_count, N);
    weight_data_dequantized.create(K, N);

    weight_data_quantized.fill<unsigned char>(0);

    const int qmax = (1 << (bits - 1)) - 1;
    for (int n = 0; n < N; n++)
    {
        const float* ptr = weight_data.row(n);
        float* scale_ptr = weight_data_quantize_scales.row(n);
        unsigned char* qptr = weight_data_quantized.row<unsigned char>(n);
        float* deqptr = weight_data_dequantized.row(n);

        for (int b = 0; b < block_count; b++)
        {
            const int k0 = b * block_size;
            const int max_kk = block_size < K - k0 ? block_size : K - k0;

            float absmax = 0.f;
            for (int k = 0; k < max_kk; k++)
            {
                const float v = (float)fabs(ptr[k0 + k]);
                if (v > absmax)
                    absmax = v;
            }

            const float scale = absmax == 0.f ? 1.f : (float)qmax / absmax;
            scale_ptr[b] = scale;

            for (int k = 0; k < max_kk; k++)
            {
                int q = (int)roundf(ptr[k0 + k] * scale);
                if (q > qmax) q = qmax;
                if (q < -qmax) q = -qmax;

                const unsigned int v = (unsigned int)q & ((1u << bits) - 1u);
                const int bit_offset = (k0 + k) * bits;
                for (int bit = 0; bit < bits; bit++)
                {
                    if (v & (1u << bit))
                    {
                        const int out_bit = bit_offset + bit;
                        qptr[out_bit / 8] |= (unsigned char)(1u << (out_bit % 8));
                    }
                }

                deqptr[k0 + k] = q / scale;
            }
        }
    }

}

static ncnn::ParamDict make_gemm_param(int M, int N, int K, int quantize_term)
{
    ncnn::ParamDict pd;
    pd.set(0, 1.f); // alpha
    pd.set(1, 1.f); // beta
    pd.set(2, 0); // transA
    pd.set(3, 1); // transB
    pd.set(4, 0); // constantA
    pd.set(5, 1); // constantB
    pd.set(6, 0); // constantC
    pd.set(7, M);
    pd.set(8, N);
    pd.set(9, K);
    pd.set(10, -1); // broadcast_type_C
    pd.set(11, 0); // output_N1M
    pd.set(14, 0); // output_transpose
    pd.set(18, quantize_term);

    return pd;
}

static int test_gemm_block_quant(const ncnn::Mat& A, const ncnn::Mat& B, const ncnn::Mat& C, const ncnn::ParamDict& pd, int bits, int block_size, int has_input_scale, int constantC)
{
    const int K = B.w;
    const int N = B.h;

    ncnn::Mat B_input_scales;
    ncnn::Mat B1 = B;
    if (has_input_scale)
    {
        B_input_scales = make_input_scales(K);
        B1 = B.clone();

        const float* input_scale_ptr = B_input_scales;
        for (int n = 0; n < N; n++)
        {
            float* ptr = B1.row(n);
            for (int k = 0; k < K; k++)
                ptr[k] /= input_scale_ptr[k];
        }
    }

    ncnn::Mat B_quantized;
    ncnn::Mat B_quantize_scales;
    ncnn::Mat B_dequantized;
    quantize_weight(B1, bits, block_size, B_quantized, B_quantize_scales, B_dequantized);

    if (has_input_scale)
    {
        const float* input_scale_ptr = B_input_scales;
        for (int n = 0; n < N; n++)
        {
            float* ptr = B_dequantized.row(n);
            for (int k = 0; k < K; k++)
                ptr[k] *= input_scale_ptr[k];
        }
    }

    std::vector<ncnn::Mat> weights;
    weights.push_back(B_quantized);
    if (constantC)
        weights.push_back(C);
    weights.push_back(B_quantize_scales);
    if (has_input_scale)
        weights.push_back(B_input_scales);

    std::vector<ncnn::Mat> a;
    a.push_back(A);
    if (!constantC && !C.empty())
        a.push_back(C);

    ncnn::Option opt;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;

    if (bits != 8)
    {
        std::vector<ncnn::Mat> ref_weights;
        ref_weights.push_back(B_dequantized);
        if (constantC)
            ref_weights.push_back(C);

        ncnn::ParamDict ref_pd = pd;
        ref_pd.set(18, 0);

        std::vector<ncnn::Mat> refs;
        test_layer_naive(ncnn::layer_to_index("Gemm"), ref_pd, ref_weights, a, 1, refs, TEST_LAYER_DISABLE_GPU_TESTING);

        for (int t = 0; t < 2; t++)
        {
            std::vector<ncnn::Mat> outputs;
            const int flag = TEST_LAYER_DISABLE_GPU_TESTING | (t ? TEST_LAYER_ENABLE_THREADING : 0);
            test_layer_cpu(ncnn::layer_to_index("Gemm"), pd, weights, opt, a, 1, outputs, std::vector<ncnn::Mat>(), flag);
            if (CompareMat(outputs, refs, 0.001f) != 0)
                return -1;
        }

        return 0;
    }

    return 0
           || test_layer_opt("Gemm", pd, weights, opt, a, 1, 0.001f, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_layer_opt("Gemm", pd, weights, opt, a, 1, 0.001f, TEST_LAYER_DISABLE_GPU_TESTING | TEST_LAYER_ENABLE_THREADING);
}

static int test_gemm(int M, int N, int K, int bits, int block_size, int has_input_scale = 0, int transA = 0, int output_transpose = 0, int output_N1M = 0, int dims3 = 0)
{
    ncnn::Mat A = dims3 || output_N1M ? ncnn::Mat(K, 1, M) : transA ? ncnn::Mat(M, K) : ncnn::Mat(K, M);
    ncnn::Mat B(K, N);
    ncnn::Mat input_scales;
    if (bits == 8)
    {
        if (has_input_scale)
            input_scales = make_input_scales(K);
        RandomizeA(A, transA, block_size, input_scales);
        RandomizeB(B, block_size, input_scales);
    }
    else
    {
        Randomize(A, -2.f, 2.f);
        Randomize(B, -2.f, 2.f);
    }

    ncnn::ParamDict pd = make_gemm_param(M, N, K, weight_block_quantize_term(bits, block_size, has_input_scale));
    pd.set(2, transA);
    pd.set(11, output_N1M);
    pd.set(14, output_transpose);
    const int ret = test_gemm_block_quant(A, B, ncnn::Mat(), pd, bits, block_size, has_input_scale, 0);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm failed M=%d N=%d K=%d bits=%d block_size=%d has_input_scale=%d transA=%d output_transpose=%d output_N1M=%d dims3=%d\n", M, N, K, bits, block_size, has_input_scale, transA, output_transpose, output_N1M, dims3);
    }

    return ret;
}

static int test_gemm_bias(int M, int N, int K, int bits, int block_size, const ncnn::Mat& C, float alpha, float beta, int has_input_scale, int transA, int output_transpose, int constantC, int output_N1M = 0)
{
    int broadcast_type_C = 0;
    if (C.dims == 1 && C.w == 1)
    {
        // scalar
        broadcast_type_C = 0;
    }
    if (C.dims == 1 && C.w == M)
    {
        // M
        // auto broadcast from h to w is the ncnn-style convention
        broadcast_type_C = 1;
    }
    if (C.dims == 1 && C.w == N)
    {
        // N
        broadcast_type_C = 4;
    }
    if (C.dims == 2 && C.w == 1 && C.h == M)
    {
        // Mx1
        broadcast_type_C = 2;
    }
    if (C.dims == 2 && C.w == N && C.h == M)
    {
        // MxN
        broadcast_type_C = 3;
    }
    if (C.dims == 2 && C.w == N && C.h == 1)
    {
        // 1xN
        broadcast_type_C = 4;
    }

    ncnn::Mat A = output_N1M ? ncnn::Mat(K, 1, M) : transA ? ncnn::Mat(M, K) : ncnn::Mat(K, M);
    ncnn::Mat B(K, N);
    ncnn::Mat input_scales;
    if (bits == 8)
    {
        if (has_input_scale)
            input_scales = make_input_scales(K);
        RandomizeA(A, transA, block_size, input_scales);
        RandomizeB(B, block_size, input_scales);
    }
    else
    {
        Randomize(A, -2.f, 2.f);
        Randomize(B, -2.f, 2.f);
    }

    ncnn::ParamDict pd = make_gemm_param(M, N, K, weight_block_quantize_term(bits, block_size, has_input_scale));
    pd.set(0, alpha);
    pd.set(1, beta);
    pd.set(2, transA);
    pd.set(6, constantC);
    pd.set(10, broadcast_type_C);
    pd.set(11, output_N1M);
    pd.set(14, output_transpose);
    const int ret = test_gemm_block_quant(A, B, C, pd, bits, block_size, has_input_scale, constantC);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm_bias failed M=%d N=%d K=%d bits=%d block_size=%d C.dims=%d C=(%d %d) alpha=%f beta=%f has_input_scale=%d transA=%d output_transpose=%d constantC=%d output_N1M=%d\n", M, N, K, bits, block_size, C.dims, C.w, C.h, alpha, beta, has_input_scale, transA, output_transpose, constantC, output_N1M);
    }

    return ret;
}

static int test_gemm_wq_int8_zero(int zero_A, int zero_B_block)
{
    const int M = 31;
    const int N = 16;
    const int K = 67;
    const int block_size = 32;

    ncnn::Mat A(K, M);
    ncnn::Mat B(K, N);
    RandomizeA(A, 0, block_size, ncnn::Mat());
    RandomizeB(B, block_size, ncnn::Mat());

    if (zero_A)
        A.fill(0.f);
    if (zero_B_block)
    {
        for (int n = 0; n < N; n++)
        {
            float* ptr = B.row(n);
            for (int k = 0; k < block_size; k++)
                ptr[k] = 0.f;
        }
    }

    const ncnn::ParamDict pd = make_gemm_param(M, N, K, weight_block_quantize_term(8, block_size));
    const int ret = test_gemm_block_quant(A, B, ncnn::Mat(), pd, 8, block_size, 0, 0);
    if (ret != 0)
        fprintf(stderr, "test_gemm_wq_int8_zero failed zero_A=%d zero_B_block=%d\n", zero_A, zero_B_block);

    return ret;
}

static int test_gemm_wq_int8_tile(int M, int N, int K, int block_size, int TILE_M, int TILE_N, int TILE_K, int has_input_scale = 0, int transA = 0, int output_transpose = 0)
{
    ncnn::Mat A = transA ? ncnn::Mat(M, K) : ncnn::Mat(K, M);
    ncnn::Mat B(K, N);
    ncnn::Mat input_scales = has_input_scale ? make_input_scales(K) : ncnn::Mat();
    RandomizeA(A, transA, block_size, input_scales);
    RandomizeB(B, block_size, input_scales);

    ncnn::ParamDict pd = make_gemm_param(M, N, K, weight_block_quantize_term(8, block_size, has_input_scale));
    pd.set(2, transA);
    pd.set(14, output_transpose);
    pd.set(20, TILE_M);
    pd.set(21, TILE_N);
    pd.set(22, TILE_K);

    const int ret = test_gemm_block_quant(A, B, ncnn::Mat(), pd, 8, block_size, has_input_scale, 0);
    if (ret != 0)
        fprintf(stderr, "test_gemm_wq_int8_tile failed M=%d N=%d K=%d TILE_M=%d TILE_N=%d TILE_K=%d input_scale=%d transA=%d output_transpose=%d\n", M, N, K, TILE_M, TILE_N, TILE_K, has_input_scale, transA, output_transpose);

    return ret;
}

static int test_gemm_wq_int8_exact_weight_tail(int K)
{
    const int M = 3;
    const int N = 8;
    const int block_size = 32;
    const int block_count = (K + block_size - 1) / block_size;

    ncnn::Mat A(K, M);
    RandomizeA(A, 0, block_size, ncnn::Mat());

    std::vector<signed char> B_storage((size_t)K * N);
    for (int n = 0; n < N; n++)
    {
        signed char* ptr = &B_storage[(size_t)n * K];
        for (int k = 0; k < K; k++)
            ptr[k] = (signed char)((n * 19 + k * 11) % 101 - 50);
    }

    ncnn::Mat B_quantized(K, N, (void*)&B_storage[0], (size_t)1u);
    ncnn::Mat B_quantize_scales(block_count, N);
    B_quantize_scales.fill(32.f);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = B_quantized;
    weights[1] = B_quantize_scales;

    ncnn::Option opt;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;

    const ncnn::ParamDict pd = make_gemm_param(M, N, K, weight_block_quantize_term(8, block_size));
    const int ret = test_layer_opt("Gemm", pd, weights, opt, A, 0.001f, TEST_LAYER_DISABLE_GPU_TESTING);
    if (ret != 0)
        fprintf(stderr, "test_gemm_wq_int8_exact_weight_tail failed K=%d\n", K);

    return ret;
}

static int test_gemm_wq_int8_input_scale_equivalence(int transA, int output_transpose)
{
    const int M = 17;
    const int N = 19;
    const int K = 67;
    const int block_size = 32;
    const int block_count = (K + block_size - 1) / block_size;

    ncnn::Mat input_scales = make_input_scales(K);
    ncnn::Mat A_scaled = transA ? ncnn::Mat(M, K) : ncnn::Mat(K, M);
    RandomizeA(A_scaled, transA, block_size, ncnn::Mat());

    ncnn::Mat A = A_scaled.clone();
    const float* input_scale_ptr = input_scales;
    for (int i = 0; i < M; i++)
    {
        for (int k = 0; k < K; k++)
        {
            if (transA)
                A.row(k)[i] /= input_scale_ptr[k];
            else
                A.row(i)[k] /= input_scale_ptr[k];
        }
    }

    for (int k = block_size; k < block_size * 2; k++)
    {
        if (transA)
        {
            A_scaled.row(k)[M - 1] = 0.f;
            A.row(k)[M - 1] = 0.f;
        }
        else
        {
            A_scaled.row(M - 1)[k] = 0.f;
            A.row(M - 1)[k] = 0.f;
        }
    }

    ncnn::Mat B_quantized(K, N, (size_t)1u);
    for (int n = 0; n < N; n++)
    {
        signed char* ptr = B_quantized.row<signed char>(n);
        for (int k = 0; k < K; k++)
            ptr[k] = (signed char)((n * 13 + k * 17) % 127 - 63);
    }

    ncnn::Mat B_quantize_scales(block_count, N);
    for (int n = 0; n < N; n++)
    {
        float* ptr = B_quantize_scales.row(n);
        for (int g = 0; g < block_count; g++)
            ptr[g] = 32.f + (n + g) % 4 * 8.f;
    }

    std::vector<ncnn::Mat> weights(2);
    weights[0] = B_quantized;
    weights[1] = B_quantize_scales;

    std::vector<ncnn::Mat> scaled_weights(3);
    scaled_weights[0] = B_quantized;
    scaled_weights[1] = B_quantize_scales;
    scaled_weights[2] = input_scales;

    ncnn::Option opt;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;

    ncnn::ParamDict pd = make_gemm_param(M, N, K, weight_block_quantize_term(8, block_size));
    pd.set(2, transA);
    pd.set(14, output_transpose);
    ncnn::ParamDict scaled_pd = make_gemm_param(M, N, K, weight_block_quantize_term(8, block_size, 1));
    scaled_pd.set(2, transA);
    scaled_pd.set(14, output_transpose);

    for (int t = 0; t < 2; t++)
    {
        ncnn::Mat output;
        ncnn::Mat scaled_output;
        const int flag = TEST_LAYER_DISABLE_GPU_TESTING | (t ? TEST_LAYER_ENABLE_THREADING : 0);
        test_layer_cpu(ncnn::layer_to_index("Gemm"), pd, weights, opt, A_scaled, output, ncnn::Mat(), flag);
        test_layer_cpu(ncnn::layer_to_index("Gemm"), scaled_pd, scaled_weights, opt, A, scaled_output, ncnn::Mat(), flag);
        if (CompareMat(output, scaled_output, 0.001f) != 0)
        {
            fprintf(stderr, "test_gemm_wq_int8_input_scale_equivalence failed transA=%d output_transpose=%d threading=%d\n", transA, output_transpose, t);
            return -1;
        }
    }

    return 0;
}

static int test_gemm_wq_int8_pipeline()
{
    const int N = 31;
    const int K = 67;
    const int block_size = 32;

    ncnn::Mat B_input_scales = make_input_scales(K);
    ncnn::Mat B_quantized(K, N, (size_t)1u);
    ncnn::Mat B_quantize_scales(3, N);
    B_quantize_scales.fill(1.f);
    for (int n = 0; n < N; n++)
    {
        signed char* ptr = B_quantized.row<signed char>(n);
        for (int k = 0; k < K; k++)
            ptr[k] = (signed char)((n * 19 + k * 11) % 101 - 50);
    }

    std::vector<ncnn::Mat> weights(3);
    weights[0] = B_quantized;
    weights[1] = B_quantize_scales;
    weights[2] = B_input_scales;

    ncnn::Layer* gemm = ncnn::create_layer_cpu("Gemm");
    if (!gemm)
        return -1;

    const ncnn::ParamDict pd = make_gemm_param(0, N, K, weight_block_quantize_term(8, block_size, 1));
    gemm->load_param(pd);
    gemm->load_model(ncnn::ModelBinFromMatArray(weights.data()));

    ncnn::Option opt;
    opt.lightmode = false;
    opt.num_threads = 2;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;

    gemm->create_pipeline(opt);

    int test_ret = 0;
    const int test_M[] = {128, 1, 7};
    const int test_threads[] = {2, 1, 4};
    for (int t = 0; t < 3; t++)
    {
        const int M = test_M[t];
        ncnn::Mat A(K, M);
        RandomizeA(A, 0, block_size, B_input_scales);

        ncnn::Mat reference;
        test_layer_naive(ncnn::layer_to_index("Gemm"), pd, weights, A, reference, TEST_LAYER_DISABLE_GPU_TESTING);

        std::vector<ncnn::Mat> bottom_blobs(1, A);
        std::vector<ncnn::Mat> top_blobs(1);
        ncnn::Option opt1 = opt;
        opt1.num_threads = test_threads[t];
        gemm->forward(bottom_blobs, top_blobs, opt1);
        if (CompareMat(top_blobs[0], reference, 0.001f) != 0)
        {
            fprintf(stderr, "test_gemm_wq_int8_pipeline failed M=%d threads=%d\n", M, opt1.num_threads);
            test_ret = -1;
            break;
        }
    }

    gemm->destroy_pipeline(opt);
    delete gemm;

    return test_ret;
}

static int test_gemm_0()
{
    return 0
           || test_gemm(3, 5, 65, 4, 64)
           || test_gemm(3, 4, 33, 4, 32, 1)
           || test_gemm_bias(3, 4, 33, 4, 32, RandomMat(1), 1.7f, 0.3f, 0, 0, 0, 1)
           || test_gemm(4, 7, 67, 6, 64)
           || test_gemm(2, 4, 31, 6, 32, 1)
           || test_gemm(3, 4, 129, 6, 128)
           || test_gemm_bias(4, 7, 67, 6, 64, RandomMat(7, 4), 0.7f, 1.3f, 1, 0, 0, 0);
}

static int test_gemm_1(int M, int N, int K, int block_size)
{
    return 0
           || test_gemm(M, N, K, 8, block_size)
           || test_gemm(M, N, K, 8, block_size, 1, 0, 1)
           || test_gemm(M, N, K, 8, block_size, 0, 1);
}

static int test_gemm_2()
{
    const int M = 5;
    const int N = 7;
    const int K = 35;
    const int block_size = 32;

    return 0
           || test_gemm_bias(M, N, K, 8, block_size, RandomMat(1), 0.7f, 1.3f, 0, 0, 0, 0)
           || test_gemm_bias(M, N, K, 8, block_size, RandomMat(M), 1.f, 0.3f, 1, 0, 0, 1)
           || test_gemm_bias(M, N, K, 8, block_size, RandomMat(1, M), 0.7f, 1.f, 0, 0, 1, 0)
           || test_gemm_bias(M, N, K, 8, block_size, RandomMat(N, M), 1.7f, 0.3f, 1, 0, 1, 1)
           || test_gemm_bias(M, N, K, 8, block_size, RandomMat(N), 0.7f, 1.3f, 0, 1, 0, 0)
           || test_gemm_bias(M, N, K, 8, block_size, RandomMat(N, 1), 1.3f, 0.7f, 1, 0, 0, 1)
           || test_gemm_bias(M, N, K, 8, block_size, RandomMat(N, M), -0.7f, -1.3f, 1, 0, 1, 0)
           || test_gemm_bias(M, N, K, 8, block_size, RandomMat(N), 1.f, 0.f, 0, 0, 0, 0)
           || test_gemm(3, 5, 67, 8, 64, 0, 0, 0, 1)
           || test_gemm_bias(3, 5, 67, 8, 64, RandomMat(5, 3), 0.7f, 0.3f, 1, 0, 1, 0, 1)
           || test_gemm_bias(17, 19, 35, 8, 32, RandomMat(19, 17), -0.7f, -1.3f, 1, 0, 0, 0)
           || test_gemm_bias(17, 19, 35, 8, 32, RandomMat(19), 1.7f, 0.3f, 1, 0, 1, 1);
}

static int test_gemm_3()
{
    return 0
           || test_gemm_wq_int8_zero(1, 0)
           || test_gemm_wq_int8_zero(0, 1)
           || test_gemm(5, 3, 65, 8, 64, 0, 0, 0, 0, 1)
           || test_gemm(7, 9, 67, 8, 64, 1, 1, 1)
           || test_gemm_wq_int8_tile(1, 17, 33, 32, 1, 4, 32)
           || test_gemm_wq_int8_tile(17, 19, 35, 32, 9, 4, 32, 1, 0, 1)
           || test_gemm_wq_int8_tile(19, 17, 67, 64, 9, 4, 64, 0, 1)
           || test_gemm_wq_int8_tile(13, 19, 35, 32, 7, 5, 17)
           || test_gemm_wq_int8_tile(7, 9, 67, 64, 5, 7, 48);
}
#endif // NCNN_WEIGHT_QUANT

int main()
{
    SRAND(7767517);

#if NCNN_WEIGHT_QUANT
    int ret = test_gemm_wq_int8_exact_weight_tail(33)
              || test_gemm_wq_int8_exact_weight_tail(34)
              || test_gemm_wq_int8_exact_weight_tail(35)
              || test_gemm_wq_int8_input_scale_equivalence(0, 0)
              || test_gemm_wq_int8_input_scale_equivalence(1, 1)
              || test_gemm_wq_int8_pipeline()
              || test_gemm_0()
              || test_gemm_2()
              || test_gemm_3();
    if (ret != 0)
        return ret;

    const int mnkb[][4] = {
        {1, 7, 31, 32},
        {2, 3, 32, 32},
        {3, 6, 33, 32},
        {8, 3, 34, 32},
        {16, 16, 35, 32},
        {17, 31, 64, 32},
        {31, 17, 65, 64},
        {3, 129, 67, 64},
        {8, 16, 128, 128},
        {19, 3, 129, 128}
    };

    for (int i = 0; i < (int)(sizeof(mnkb) / sizeof(mnkb[0])); i++)
    {
        ret = test_gemm_1(mnkb[i][0], mnkb[i][1], mnkb[i][2], mnkb[i][3]);
        if (ret != 0)
            return ret;
    }

#endif

    return 0;
}
