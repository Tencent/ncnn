// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#if NCNN_WEIGHT_QUANT
static void pack_signed_weight(unsigned char* ptr, int k, int bits, int q)
{
    const unsigned int v = (unsigned int)q & ((1u << bits) - 1);
    const int bit_offset = k * bits;

    for (int b = 0; b < bits; b++)
    {
        if (v & (1u << b))
        {
            const int out_bit = bit_offset + b;
            ptr[out_bit / 8] |= (unsigned char)(1u << (out_bit % 8));
        }
    }
}

static int float2int_weight(float v, int bits)
{
    const int qmax = (1 << (bits - 1)) - 1;
    int q = (int)roundf(v);
    if (q > qmax) q = qmax;
    if (q < -qmax) q = -qmax;
    return q;
}

static int weight_block_quantize_term(int bits, int block_size, int has_input_scale)
{
    const int block_size_code = block_size == 32 ? 0 : block_size == 64 ? 1 : 2;
    return bits * 100 + (has_input_scale ? 10 : 0) + block_size_code;
}

static ncnn::Mat make_input_scales(int size, int offset)
{
    ncnn::Mat scales(size);
    float* ptr = scales;
    const float scale_table[5] = {0.5f, 1.f, 2.f, 0.25f, 4.f};
    for (int i = 0; i < size; i++)
        ptr[i] = scale_table[(i + offset) % 5];

    return scales;
}

static ncnn::Mat make_w8a8_mat(int width, int height, int block_size, const ncnn::Mat& input_scales = ncnn::Mat())
{
    ncnn::Mat m(width, height);
    const float* scale_ptr = input_scales;

    for (int y = 0; y < height; y++)
    {
        float* ptr = m.row(y);
        for (int x = 0; x < width; x++)
        {
            int q = RandomInt(-120, 121);
            if (x % block_size == 0)
                q = y % 2 == 0 ? 127 : -127;

            ptr[x] = scale_ptr ? q / (64.f * scale_ptr[x]) : q / 64.f;
        }
    }

    return m;
}

static ncnn::Mat make_w8a8_cache(int width, int height)
{
    ncnn::Mat m(width, height);
    std::vector<float> values(width);
    for (int x = 0; x < width; x++)
        values[x] = RandomInt(-120, 121) / 64.f;

    for (int y = 0; y < height; y++)
    {
        float* ptr = m.row(y);
        for (int x = 0; x < width; x++)
            ptr[x] = values[x];
    }

    return m;
}

static int quantize_weight(const ncnn::Mat& weight_data, int bits, int block_size, const ncnn::Mat& input_scales, ncnn::Mat& weight_data_quantized, ncnn::Mat& weight_data_quantize_scales, ncnn::Mat& weight_data_dequantized)
{
    const int K = weight_data.w;
    const int N = weight_data.h;
    const int block_count = (K + block_size - 1) / block_size;

    weight_data_quantized.create((K * bits + 7) / 8, N, (size_t)1u);
    weight_data_quantize_scales.create(block_count, N);
    weight_data_dequantized.create(K, N);
    if (weight_data_quantized.empty() || weight_data_quantize_scales.empty() || weight_data_dequantized.empty())
        return -100;

    weight_data_quantized.fill((unsigned char)0);

    const int qmax = (1 << (bits - 1)) - 1;
    const float* input_scale_ptr = input_scales;
    for (int n = 0; n < N; n++)
    {
        const float* ptr = weight_data.row(n);
        unsigned char* qptr = weight_data_quantized.row<unsigned char>(n);
        float* scale_ptr = weight_data_quantize_scales.row(n);
        float* deqptr = weight_data_dequantized.row(n);

        for (int b = 0; b < block_count; b++)
        {
            const int k0 = b * block_size;
            const int max_kk = block_size < K - k0 ? block_size : K - k0;

            float absmax = 0.f;
            for (int k = 0; k < max_kk; k++)
            {
                const float v = fabsf(ptr[k0 + k]);
                if (v > absmax)
                    absmax = v;
            }

            const float scale = absmax == 0.f ? 1.f : qmax / absmax;
            scale_ptr[b] = scale;

            for (int k = 0; k < max_kk; k++)
            {
                const int q = float2int_weight(ptr[k0 + k] * scale, bits);
                pack_signed_weight(qptr, k0 + k, bits, q);
                deqptr[k0 + k] = q / scale * (input_scale_ptr ? input_scale_ptr[k0 + k] : 1.f);
            }
        }
    }

    return 0;
}

static int make_mha_weights(int qdim, int kdim, int vdim, int embed_dim, int bits, int block_size, int has_input_scale, std::vector<ncnn::Mat>& weights, std::vector<ncnn::Mat>& ref_weights)
{
    ncnn::Mat q_input_scales;
    ncnn::Mat k_input_scales;
    ncnn::Mat v_input_scales;
    ncnn::Mat out_input_scales;
    if (has_input_scale)
    {
        q_input_scales = make_input_scales(qdim, 0);
        k_input_scales = make_input_scales(kdim, 1);
        v_input_scales = make_input_scales(vdim, 2);
        out_input_scales = make_input_scales(embed_dim, 3);
        if (bits == 8)
            out_input_scales.fill(1.f);
    }

    ncnn::Mat q_weight = bits == 8 ? make_w8a8_mat(qdim, embed_dim, block_size) : RandomMat(qdim, embed_dim, -1.f, 1.f);
    ncnn::Mat k_weight = bits == 8 ? make_w8a8_mat(kdim, embed_dim, block_size) : RandomMat(kdim, embed_dim, -1.f, 1.f);
    ncnn::Mat v_weight = bits == 8 ? make_w8a8_mat(vdim, embed_dim, block_size) : RandomMat(vdim, embed_dim, -1.f, 1.f);
    ncnn::Mat out_weight = bits == 8 ? make_w8a8_mat(embed_dim, qdim, block_size) : RandomMat(embed_dim, qdim, -1.f, 1.f);

    if (bits == 8 && has_input_scale)
    {
        const float* ptr = v_weight.row(0);
        for (int i = 1; i < embed_dim; i++)
        {
            float* outptr = v_weight.row(i);
            for (int j = 0; j < vdim; j++)
                outptr[j] = ptr[j];
        }
    }

    ncnn::Mat q_weight_quantized;
    ncnn::Mat k_weight_quantized;
    ncnn::Mat v_weight_quantized;
    ncnn::Mat out_weight_quantized;
    ncnn::Mat q_weight_scales;
    ncnn::Mat k_weight_scales;
    ncnn::Mat v_weight_scales;
    ncnn::Mat out_weight_scales;
    ncnn::Mat q_weight_dequantized;
    ncnn::Mat k_weight_dequantized;
    ncnn::Mat v_weight_dequantized;
    ncnn::Mat out_weight_dequantized;

    if (quantize_weight(q_weight, bits, block_size, q_input_scales, q_weight_quantized, q_weight_scales, q_weight_dequantized) != 0
            || quantize_weight(k_weight, bits, block_size, k_input_scales, k_weight_quantized, k_weight_scales, k_weight_dequantized) != 0
            || quantize_weight(v_weight, bits, block_size, v_input_scales, v_weight_quantized, v_weight_scales, v_weight_dequantized) != 0
            || quantize_weight(out_weight, bits, block_size, out_input_scales, out_weight_quantized, out_weight_scales, out_weight_dequantized) != 0)
        return -100;

    ncnn::Mat q_bias = RandomMat(embed_dim, -1.f, 1.f);
    ncnn::Mat k_bias = RandomMat(embed_dim, -1.f, 1.f);
    ncnn::Mat v_bias = RandomMat(embed_dim, -1.f, 1.f);
    ncnn::Mat out_bias = RandomMat(qdim, -1.f, 1.f);
    if (bits == 8 && has_input_scale)
        v_bias.fill(0.f);

    weights.resize(has_input_scale ? 16 : 12);
    weights[0] = q_weight_quantized;
    weights[1] = q_bias;
    weights[2] = k_weight_quantized;
    weights[3] = k_bias;
    weights[4] = v_weight_quantized;
    weights[5] = v_bias;
    weights[6] = out_weight_quantized;
    weights[7] = out_bias;
    weights[8] = q_weight_scales;
    weights[9] = k_weight_scales;
    weights[10] = v_weight_scales;
    weights[11] = out_weight_scales;

    if (has_input_scale)
    {
        weights[12] = q_input_scales;
        weights[13] = k_input_scales;
        weights[14] = v_input_scales;
        weights[15] = out_input_scales;
    }

    ref_weights.resize(8);
    ref_weights[0] = q_weight_dequantized.reshape(embed_dim * qdim);
    ref_weights[1] = q_bias;
    ref_weights[2] = k_weight_dequantized.reshape(embed_dim * kdim);
    ref_weights[3] = k_bias;
    ref_weights[4] = v_weight_dequantized.reshape(embed_dim * vdim);
    ref_weights[5] = v_bias;
    ref_weights[6] = out_weight_dequantized.reshape(qdim * embed_dim);
    ref_weights[7] = out_bias;

    return 0;
}

static int test_multiheadattention_invalid_weight_block_quantize_term()
{
    const int invalid_quantize_terms[] = {403, 420, 700};

    for (int i = 0; i < 3; i++)
    {
        ncnn::ParamDict pd;
        pd.set(0, 8);
        pd.set(1, 2);
        pd.set(2, 64);
        pd.set(3, 8);
        pd.set(4, 8);
        pd.set(18, invalid_quantize_terms[i]);

        ncnn::Layer* mha = ncnn::create_layer_naive("MultiHeadAttention");
        if (!mha)
            return -100;

        const int ret = mha->load_param(pd);
        delete mha;

        if (ret == 0)
        {
            fprintf(stderr, "test_multiheadattention_invalid_weight_block_quantize_term accepted quantize_term=%d\n", invalid_quantize_terms[i]);
            return -1;
        }
    }

    return 0;
}

static int test_multiheadattention_block_quant(int qdim, int kdim, int vdim, int embed_dim, int num_heads, int bits, int block_size, int attn_mask, int has_input_scale, int zero_input_group = 0)
{
    std::vector<ncnn::Mat> weights;
    std::vector<ncnn::Mat> ref_weights;
    int ret = make_mha_weights(qdim, kdim, vdim, embed_dim, bits, block_size, has_input_scale, weights, ref_weights);
    if (ret != 0)
        return ret;

    const int src_seqlen = 5;
    const int dst_seqlen = 6;
    std::vector<ncnn::Mat> as(3);
    as[0] = bits == 8 ? make_w8a8_mat(qdim, src_seqlen, block_size, has_input_scale ? weights[12] : ncnn::Mat()) : RandomMat(qdim, src_seqlen, -1.f, 1.f);
    as[1] = bits == 8 ? make_w8a8_mat(kdim, dst_seqlen, block_size, has_input_scale ? weights[13] : ncnn::Mat()) : RandomMat(kdim, dst_seqlen, -1.f, 1.f);
    as[2] = bits == 8 ? make_w8a8_mat(vdim, dst_seqlen, block_size, has_input_scale ? weights[14] : ncnn::Mat()) : RandomMat(vdim, dst_seqlen, -1.f, 1.f);

    if (zero_input_group)
    {
        const int q_zero = qdim < block_size ? qdim : block_size;
        const int k_zero = kdim < block_size ? kdim : block_size;
        const int v_zero = vdim < block_size ? vdim : block_size;
        for (int i = 0; i < src_seqlen; i++)
            for (int j = 0; j < q_zero; j++)
                as[0].row(i)[j] = 0.f;
        for (int i = 0; i < dst_seqlen; i++)
        {
            for (int j = 0; j < k_zero; j++)
                as[1].row(i)[j] = 0.f;
            for (int j = 0; j < v_zero; j++)
                as[2].row(i)[j] = 0.f;
        }
    }

    if (attn_mask)
        as.push_back(RandomMat(dst_seqlen, src_seqlen, -1.f, 0.f));

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, kdim);
    pd.set(4, vdim);
    pd.set(5, attn_mask);
    pd.set(6, 0.7f / sqrtf(embed_dim / num_heads));
    pd.set(18, weight_block_quantize_term(bits, block_size, has_input_scale));

    ncnn::Option opt;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;

    if (bits != 8)
    {
        ncnn::ParamDict ref_pd = pd;
        ref_pd.set(18, 0);

        std::vector<ncnn::Mat> refs;
        ret = test_layer_naive(ncnn::layer_to_index("MultiHeadAttention"), ref_pd, ref_weights, as, 1, refs, TEST_LAYER_DISABLE_GPU_TESTING);

        for (int t = 0; t < 2 && ret == 0; t++)
        {
            std::vector<ncnn::Mat> outputs;
            const int flags = TEST_LAYER_DISABLE_GPU_TESTING | (t ? TEST_LAYER_ENABLE_THREADING : 0);
            ret = test_layer_cpu(ncnn::layer_to_index("MultiHeadAttention"), pd, weights, opt, as, 1, outputs, std::vector<ncnn::Mat>(), flags);
            if (ret == 0)
                ret = CompareMat(outputs, refs, 0.001f);
        }
    }
    else
    {
        for (int t = 0; t < 2 && ret == 0; t++)
        {
            const int flags = TEST_LAYER_DISABLE_GPU_TESTING | (t ? TEST_LAYER_ENABLE_THREADING : 0);
            ret = test_layer_opt("MultiHeadAttention", pd, weights, opt, as, 1, 0.001f, flags);
        }
    }
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_block_quant failed qdim=%d kdim=%d vdim=%d embed_dim=%d heads=%d bits=%d block=%d mask=%d input_scale=%d zero=%d\n", qdim, kdim, vdim, embed_dim, num_heads, bits, block_size, attn_mask, has_input_scale, zero_input_group);
    }

    return ret;
}

static int test_multiheadattention_block_quant_kvcache(int bits, int block_size, int attn_mask, int has_input_scale)
{
    const int qdim = 10;
    const int embed_dim = 8;
    const int src_seqlen = bits == 8 ? 1 : 3;

    std::vector<ncnn::Mat> weights;
    std::vector<ncnn::Mat> ref_weights;
    int ret = make_mha_weights(qdim, qdim, qdim, embed_dim, bits, block_size, has_input_scale, weights, ref_weights);
    if (ret != 0)
        return ret;

    std::vector<ncnn::Mat> as(attn_mask ? 4 : 3);
    as[0] = bits == 8 ? make_w8a8_mat(qdim, src_seqlen, block_size, has_input_scale ? weights[12] : ncnn::Mat()) : RandomMat(qdim, src_seqlen, -1.f, 1.f);
    if (attn_mask)
    {
        as[1] = RandomMat(5 + src_seqlen, src_seqlen, -1.f, 0.f);
        as[2] = RandomMat(5, embed_dim, -1.f, 1.f);
        as[3] = bits == 8 && has_input_scale ? make_w8a8_cache(5, embed_dim) : RandomMat(5, embed_dim, -1.f, 1.f);
    }
    else
    {
        as[1] = RandomMat(5, embed_dim, -1.f, 1.f);
        as[2] = bits == 8 && has_input_scale ? make_w8a8_cache(5, embed_dim) : RandomMat(5, embed_dim, -1.f, 1.f);
    }

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, 2);
    pd.set(2, embed_dim * qdim);
    pd.set(3, qdim);
    pd.set(4, qdim);
    pd.set(5, attn_mask);
    pd.set(6, 0.7f / sqrtf(4.f));
    pd.set(7, 1);
    pd.set(18, weight_block_quantize_term(bits, block_size, has_input_scale));

    ncnn::Option opt;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;

    if (bits != 8)
    {
        ncnn::ParamDict ref_pd = pd;
        ref_pd.set(18, 0);

        std::vector<ncnn::Mat> refs;
        ret = test_layer_naive(ncnn::layer_to_index("MultiHeadAttention"), ref_pd, ref_weights, as, 3, refs, TEST_LAYER_DISABLE_GPU_TESTING);

        for (int t = 0; t < 2 && ret == 0; t++)
        {
            std::vector<ncnn::Mat> outputs;
            const int flags = TEST_LAYER_DISABLE_GPU_TESTING | (t ? TEST_LAYER_ENABLE_THREADING : 0);
            ret = test_layer_cpu(ncnn::layer_to_index("MultiHeadAttention"), pd, weights, opt, as, 3, outputs, std::vector<ncnn::Mat>(), flags);
            if (ret == 0)
                ret = CompareMat(outputs, refs, 0.001f);
        }
    }
    else
    {
        for (int t = 0; t < 2 && ret == 0; t++)
        {
            const int flags = TEST_LAYER_DISABLE_GPU_TESTING | (t ? TEST_LAYER_ENABLE_THREADING : 0);
            ret = test_layer_opt("MultiHeadAttention", pd, weights, opt, as, 3, 0.001f, flags);
        }
    }

    if (ret != 0)
        fprintf(stderr, "test_multiheadattention_block_quant_kvcache failed bits=%d block=%d mask=%d input_scale=%d\n", bits, block_size, attn_mask, has_input_scale);

    return ret;
}

static int test_multiheadattention_block_quant_cross_kvcache(int bits, int block_size, int attn_mask, int has_input_scale)
{
    const int qdim = 65;
    const int kdim = 33;
    const int vdim = 49;
    const int embed_dim = 64;
    const int src_seqlen = bits == 8 ? 1 : 3;

    std::vector<ncnn::Mat> weights;
    std::vector<ncnn::Mat> ref_weights;
    int ret = make_mha_weights(qdim, kdim, vdim, embed_dim, bits, block_size, has_input_scale, weights, ref_weights);
    if (ret != 0)
        return ret;

    std::vector<ncnn::Mat> as(attn_mask ? 6 : 5);
    as[0] = bits == 8 ? make_w8a8_mat(qdim, src_seqlen, block_size, has_input_scale ? weights[12] : ncnn::Mat()) : RandomMat(qdim, src_seqlen, -1.f, 1.f);
    as[1] = bits == 8 ? make_w8a8_mat(kdim, 2, block_size, has_input_scale ? weights[13] : ncnn::Mat()) : RandomMat(kdim, 2, -1.f, 1.f);
    as[2] = bits == 8 ? make_w8a8_mat(vdim, 2, block_size, has_input_scale ? weights[14] : ncnn::Mat()) : RandomMat(vdim, 2, -1.f, 1.f);
    if (attn_mask)
    {
        as[3] = RandomMat(5, src_seqlen, -1.f, 0.f);
        as[4] = RandomMat(5, embed_dim, -1.f, 1.f);
        as[5] = bits == 8 && has_input_scale ? make_w8a8_cache(5, embed_dim) : RandomMat(5, embed_dim, -1.f, 1.f);
    }
    else
    {
        as[3] = RandomMat(5, embed_dim, -1.f, 1.f);
        as[4] = bits == 8 && has_input_scale ? make_w8a8_cache(5, embed_dim) : RandomMat(5, embed_dim, -1.f, 1.f);
    }

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, 4);
    pd.set(2, embed_dim * qdim);
    pd.set(3, kdim);
    pd.set(4, vdim);
    pd.set(5, attn_mask);
    pd.set(6, 0.7f / sqrtf(16.f));
    pd.set(7, 1);
    pd.set(18, weight_block_quantize_term(bits, block_size, has_input_scale));

    ncnn::Option opt;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;

    if (bits != 8)
    {
        ncnn::ParamDict ref_pd = pd;
        ref_pd.set(18, 0);

        std::vector<ncnn::Mat> refs;
        ret = test_layer_naive(ncnn::layer_to_index("MultiHeadAttention"), ref_pd, ref_weights, as, 3, refs, TEST_LAYER_DISABLE_GPU_TESTING);

        for (int t = 0; t < 2 && ret == 0; t++)
        {
            std::vector<ncnn::Mat> outputs;
            const int flags = TEST_LAYER_DISABLE_GPU_TESTING | (t ? TEST_LAYER_ENABLE_THREADING : 0);
            ret = test_layer_cpu(ncnn::layer_to_index("MultiHeadAttention"), pd, weights, opt, as, 3, outputs, std::vector<ncnn::Mat>(), flags);
            if (ret == 0)
                ret = CompareMat(outputs, refs, 0.001f);
        }
    }
    else
    {
        for (int t = 0; t < 2 && ret == 0; t++)
        {
            const int flags = TEST_LAYER_DISABLE_GPU_TESTING | (t ? TEST_LAYER_ENABLE_THREADING : 0);
            ret = test_layer_opt("MultiHeadAttention", pd, weights, opt, as, 3, 0.001f, flags);
        }
    }

    if (ret != 0)
        fprintf(stderr, "test_multiheadattention_block_quant_cross_kvcache failed bits=%d block=%d mask=%d input_scale=%d\n", bits, block_size, attn_mask, has_input_scale);

    return ret;
}

static int test_multiheadattention_block_quant_pipeline()
{
    const int qdim = 35;
    const int embed_dim = 32;
    const int block_size = 32;

    std::vector<ncnn::Mat> weights;
    std::vector<ncnn::Mat> ref_weights;
    int ret = make_mha_weights(qdim, qdim, qdim, embed_dim, 8, block_size, 1, weights, ref_weights);
    if (ret != 0)
        return ret;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, 4);
    pd.set(2, embed_dim * qdim);
    pd.set(3, qdim);
    pd.set(4, qdim);
    pd.set(6, 0.7f / sqrtf(8.f));
    pd.set(7, 1);
    pd.set(18, weight_block_quantize_term(8, block_size, 1));

    ncnn::Option opt;
    opt.lightmode = false;
    opt.num_threads = 2;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;

    ncnn::Layer* mha = ncnn::create_layer_cpu("MultiHeadAttention");
    if (!mha)
        return -100;

    ret = mha->load_param(pd);
    if (ret == 0)
        ret = mha->load_model(ncnn::ModelBinFromMatArray(weights.data()));
    if (ret == 0)
        ret = mha->create_pipeline(opt);
    if (ret != 0)
    {
        delete mha;
        return ret;
    }

    int test_ret = 0;
    std::vector<ncnn::Mat> prefill_inputs(3);
    prefill_inputs[0] = make_w8a8_mat(qdim, 4, block_size, weights[12]);

    std::vector<ncnn::Mat> prefill_reference;
    std::vector<ncnn::Mat> prefill_outputs(3);
    test_ret = test_layer_naive(ncnn::layer_to_index("MultiHeadAttention"), pd, weights, prefill_inputs, 3, prefill_reference, TEST_LAYER_DISABLE_GPU_TESTING);
    if (test_ret == 0)
        test_ret = mha->forward(prefill_inputs, prefill_outputs, opt);
    if (test_ret == 0)
        test_ret = CompareMat(prefill_outputs, prefill_reference, 0.001f);

    for (int i = 0; test_ret == 0 && i < 3; i++)
    {
        if (prefill_outputs[i].elembits() != 32 || prefill_outputs[i].elempack != 1)
            test_ret = -1;
    }
    if (test_ret == 0 && (prefill_outputs[1].w != 4 || prefill_outputs[2].w != 4))
        test_ret = -1;

    std::vector<ncnn::Mat> decode_reference_inputs(3);
    std::vector<ncnn::Mat> decode_inputs(3);
    decode_inputs[0] = make_w8a8_mat(qdim, 1, block_size, weights[12]);
    decode_reference_inputs[0] = decode_inputs[0];
    if (test_ret == 0)
    {
        decode_reference_inputs[1] = prefill_reference[1];
        decode_reference_inputs[2] = prefill_reference[2];
        decode_inputs[1] = prefill_outputs[1];
        decode_inputs[2] = prefill_outputs[2];
    }

    std::vector<ncnn::Mat> decode_reference;
    std::vector<ncnn::Mat> decode_outputs(3);
    if (test_ret == 0)
        test_ret = test_layer_naive(ncnn::layer_to_index("MultiHeadAttention"), pd, weights, decode_reference_inputs, 3, decode_reference, TEST_LAYER_DISABLE_GPU_TESTING);
    ncnn::Option decode_opt = opt;
    decode_opt.num_threads = 4;
    if (test_ret == 0)
        test_ret = mha->forward(decode_inputs, decode_outputs, decode_opt);
    if (test_ret == 0)
        test_ret = CompareMat(decode_outputs, decode_reference, 0.001f);

    for (int i = 0; test_ret == 0 && i < 3; i++)
    {
        if (decode_outputs[i].elembits() != 32 || decode_outputs[i].elempack != 1)
            test_ret = -1;
    }
    if (test_ret == 0 && (decode_outputs[1].w != 5 || decode_outputs[2].w != 5))
        test_ret = -1;

    const int destroy_ret = mha->destroy_pipeline(opt);
    delete mha;

    if (test_ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_block_quant_pipeline failed ret=%d\n", test_ret);
        return test_ret;
    }

    return destroy_ret;
}

static int test_multiheadattention_block_quant_0()
{
    return 0
           || test_multiheadattention_block_quant(13, 9, 11, 8, 2, 4, 32, 0, 0)
           || test_multiheadattention_block_quant(10, 10, 10, 8, 2, 6, 64, 1, 0)
           || test_multiheadattention_block_quant(12, 7, 9, 8, 2, 8, 128, 0, 0)
           || test_multiheadattention_block_quant(35, 33, 31, 32, 4, 8, 32, 1, 0)
           || test_multiheadattention_block_quant(35, 33, 31, 72, 8, 8, 32, 0, 0)
           || test_multiheadattention_block_quant(65, 33, 49, 64, 4, 8, 64, 0, 1)
           || test_multiheadattention_block_quant(129, 129, 129, 128, 8, 8, 128, 1, 1)
           || test_multiheadattention_block_quant(13, 9, 11, 8, 2, 4, 64, 1, 1)
           || test_multiheadattention_block_quant(35, 33, 31, 32, 4, 8, 32, 0, 0, 1);
}

static int test_multiheadattention_block_quant_1()
{
    return 0
           || test_multiheadattention_block_quant_kvcache(4, 64, 0, 0)
           || test_multiheadattention_block_quant_kvcache(4, 32, 1, 1)
           || test_multiheadattention_block_quant_kvcache(8, 32, 0, 0)
           || test_multiheadattention_block_quant_kvcache(8, 64, 1, 1);
}

static int test_multiheadattention_block_quant_2()
{
    return 0
           || test_multiheadattention_block_quant_cross_kvcache(6, 32, 0, 0)
           || test_multiheadattention_block_quant_cross_kvcache(6, 64, 1, 1)
           || test_multiheadattention_block_quant_cross_kvcache(8, 32, 0, 0)
           || test_multiheadattention_block_quant_cross_kvcache(8, 128, 1, 1);
}

#endif // NCNN_WEIGHT_QUANT

int main()
{
    SRAND(7767517);

#if NCNN_WEIGHT_QUANT
    return 0
           || test_multiheadattention_invalid_weight_block_quantize_term()
           || test_multiheadattention_block_quant_0()
           || test_multiheadattention_block_quant_1()
           || test_multiheadattention_block_quant_2()
           || test_multiheadattention_block_quant_pipeline();
#else
    ncnn::ParamDict pd;
    pd.set(0, 4);
    pd.set(1, 2);
    pd.set(2, 20);
    pd.set(3, 5);
    pd.set(4, 5);
    pd.set(18, 410);

    ncnn::Layer* mha = ncnn::create_layer_naive("MultiHeadAttention");
    if (!mha)
        return -100;

    const int ret = mha->load_param(pd);
    delete mha;

    if (ret == 0)
    {
        fprintf(stderr, "test_multiheadattention_block_quant failed NCNN_WEIGHT_QUANT=OFF accepted weight block quantization\n");
        return -1;
    }

    return 0;
#endif
}
