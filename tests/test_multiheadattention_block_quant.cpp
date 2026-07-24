// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#if NCNN_WEIGHT_QUANT
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

static ncnn::Mat RandomWQInt8Mat(int width, int height, int block_size, const ncnn::Mat& input_scales = ncnn::Mat())
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

static ncnn::Mat RandomWQInt8Cache(int width, int height)
{
    ncnn::Mat m(width, height);

    // keep all channels identical so output projection quantization is deterministic
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

static void quantize_weight(const ncnn::Mat& weight_data, int bits, int block_size, const ncnn::Mat& input_scales, ncnn::Mat& weight_data_quantized, ncnn::Mat& weight_data_quantize_scales, ncnn::Mat& weight_data_dequantized)
{
    const int K = weight_data.w;
    const int N = weight_data.h;
    const int block_count = (K + block_size - 1) / block_size;

    weight_data_quantized.create((K * bits + 7) / 8, N, (size_t)1u);
    weight_data_quantize_scales.create(block_count, N);
    weight_data_dequantized.create(K, N);

    weight_data_quantized.fill<unsigned char>(0);

    const int qmax = (1 << (bits - 1)) - 1;
    const float* input_scale_ptr = input_scales;
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

                deqptr[k0 + k] = q / scale * (input_scale_ptr ? input_scale_ptr[k0 + k] : 1.f);
            }
        }
    }
}

static void make_mha_weights(int qdim, int kdim, int vdim, int embed_dim, int bits, int block_size, int has_input_scale, std::vector<ncnn::Mat>& weights, std::vector<ncnn::Mat>& ref_weights)
{
    const int weight_w[4] = {qdim, kdim, vdim, embed_dim};
    const int weight_h[4] = {embed_dim, embed_dim, embed_dim, qdim};

    ncnn::Mat input_scales[4];
    if (has_input_scale)
    {
        for (int i = 0; i < 3; i++)
            input_scales[i] = make_input_scales(weight_w[i], i);

        input_scales[3].create(embed_dim);
        float* out_scale_ptr = input_scales[3];
        const float out_scale_table[5] = {1.f, 2.f, 3.f, 5.f, 8.f};
        for (int i = 0; i < embed_dim; i++)
            out_scale_ptr[i] = out_scale_table[i % 5];
    }

    ncnn::Mat weight_data[4];
    for (int i = 0; i < 4; i++)
    {
        if (bits == 8)
            weight_data[i] = RandomWQInt8Mat(weight_w[i], weight_h[i], block_size);
        else
            weight_data[i] = RandomMat(weight_w[i], weight_h[i], -1.f, 1.f);
    }

    if (bits == 8)
    {
        // keep the output projection dynamic quantization away from half-integer rounding boundaries
        for (int p = 0; p < 3; p++)
        {
            const float* ptr = weight_data[p].row(0);
            for (int i = 1; i < embed_dim; i++)
            {
                float* outptr = weight_data[p].row(i);
                for (int j = 0; j < weight_w[p]; j++)
                    outptr[j] = ptr[j];
            }
        }
    }

    ncnn::Mat weight_data_quantized[4];
    ncnn::Mat weight_data_quantize_scales[4];
    ncnn::Mat weight_data_dequantized[4];
    for (int i = 0; i < 4; i++)
        quantize_weight(weight_data[i], bits, block_size, input_scales[i], weight_data_quantized[i], weight_data_quantize_scales[i], weight_data_dequantized[i]);

    ncnn::Mat bias_data[4];
    for (int i = 0; i < 3; i++)
        bias_data[i] = RandomMat(embed_dim, -1.f, 1.f);
    bias_data[3] = RandomMat(qdim, -1.f, 1.f);
    if (bits == 8)
    {
        for (int i = 0; i < 3; i++)
            bias_data[i].fill(0.f);
    }

    weights.resize(has_input_scale ? 16 : 12);
    ref_weights.resize(8);
    for (int i = 0; i < 4; i++)
    {
        weights[i * 2] = weight_data_quantized[i];
        weights[i * 2 + 1] = bias_data[i];
        weights[8 + i] = weight_data_quantize_scales[i];
        if (has_input_scale)
            weights[12 + i] = input_scales[i];

        ref_weights[i * 2] = weight_data_dequantized[i].reshape(weight_w[i] * weight_h[i]);
        ref_weights[i * 2 + 1] = bias_data[i];
    }
}

static int test_multiheadattention_block_quant(const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& ref_weights, const std::vector<ncnn::Mat>& inputs, int top_blob_count, int bits)
{
    if (bits == 8)
        return test_layer("MultiHeadAttention", pd, weights, inputs, top_blob_count, 0.001f, TEST_LAYER_DISABLE_GPU_TESTING | TEST_LAYER_ENABLE_THREADING);

    ncnn::ParamDict ref_pd = pd;
    ref_pd.set(18, 0);

    ncnn::Option opt;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;

    std::vector<ncnn::Mat> refs;
    test_layer_naive(ncnn::layer_to_index("MultiHeadAttention"), ref_pd, ref_weights, inputs, top_blob_count, refs, TEST_LAYER_DISABLE_GPU_TESTING);
    for (int t = 0; t < 2; t++)
    {
        std::vector<ncnn::Mat> outputs;
        const int flags = TEST_LAYER_DISABLE_GPU_TESTING | (t ? TEST_LAYER_ENABLE_THREADING : 0);
        test_layer_cpu(ncnn::layer_to_index("MultiHeadAttention"), pd, weights, opt, inputs, top_blob_count, outputs, std::vector<ncnn::Mat>(), flags);
        if (CompareMat(outputs, refs, 0.001f) != 0)
            return -1;
    }

    return 0;
}

static int test_multiheadattention_block_quant(int qdim, int kdim, int vdim, int embed_dim, int num_heads, int bits, int block_size, int attn_mask, int has_input_scale, int zero_input_group = 0)
{
    std::vector<ncnn::Mat> weights;
    std::vector<ncnn::Mat> ref_weights;
    make_mha_weights(qdim, kdim, vdim, embed_dim, bits, block_size, has_input_scale, weights, ref_weights);

    const int src_seqlen = 5;
    const int dst_seqlen = 6;
    std::vector<ncnn::Mat> as(3);
    if (bits == 8)
    {
        const ncnn::Mat q_input_scales = has_input_scale ? weights[12] : ncnn::Mat();
        const ncnn::Mat k_input_scales = has_input_scale ? weights[13] : ncnn::Mat();
        const ncnn::Mat v_input_scales = has_input_scale ? weights[14] : ncnn::Mat();
        as[0] = RandomWQInt8Mat(qdim, src_seqlen, block_size, q_input_scales);
        as[1] = RandomWQInt8Mat(kdim, dst_seqlen, block_size, k_input_scales);
        as[2] = RandomWQInt8Mat(vdim, dst_seqlen, block_size, v_input_scales);
    }
    else
    {
        as[0] = RandomMat(qdim, src_seqlen, -1.f, 1.f);
        as[1] = RandomMat(kdim, dst_seqlen, -1.f, 1.f);
        as[2] = RandomMat(vdim, dst_seqlen, -1.f, 1.f);
    }

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

    const int ret = test_multiheadattention_block_quant(pd, weights, ref_weights, as, 1, bits);
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
    make_mha_weights(qdim, qdim, qdim, embed_dim, bits, block_size, has_input_scale, weights, ref_weights);

    std::vector<ncnn::Mat> as(attn_mask ? 4 : 3);
    if (bits == 8)
    {
        const ncnn::Mat input_scales = has_input_scale ? weights[12] : ncnn::Mat();
        as[0] = RandomWQInt8Mat(qdim, src_seqlen, block_size, input_scales);
    }
    else
    {
        as[0] = RandomMat(qdim, src_seqlen, -1.f, 1.f);
    }
    if (attn_mask)
    {
        as[1] = RandomMat(5 + src_seqlen, src_seqlen, -1.f, 0.f);
        as[2] = RandomMat(5, embed_dim, -1.f, 1.f);
        as[3] = bits == 8 && has_input_scale ? RandomWQInt8Cache(5, embed_dim) : RandomMat(5, embed_dim, -1.f, 1.f);
    }
    else
    {
        as[1] = RandomMat(5, embed_dim, -1.f, 1.f);
        as[2] = bits == 8 && has_input_scale ? RandomWQInt8Cache(5, embed_dim) : RandomMat(5, embed_dim, -1.f, 1.f);
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

    const int ret = test_multiheadattention_block_quant(pd, weights, ref_weights, as, 3, bits);

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
    make_mha_weights(qdim, kdim, vdim, embed_dim, bits, block_size, has_input_scale, weights, ref_weights);

    std::vector<ncnn::Mat> as(attn_mask ? 6 : 5);
    if (bits == 8)
    {
        const ncnn::Mat q_input_scales = has_input_scale ? weights[12] : ncnn::Mat();
        const ncnn::Mat k_input_scales = has_input_scale ? weights[13] : ncnn::Mat();
        const ncnn::Mat v_input_scales = has_input_scale ? weights[14] : ncnn::Mat();
        as[0] = RandomWQInt8Mat(qdim, src_seqlen, block_size, q_input_scales);
        as[1] = RandomWQInt8Mat(kdim, 2, block_size, k_input_scales);
        as[2] = RandomWQInt8Mat(vdim, 2, block_size, v_input_scales);
    }
    else
    {
        as[0] = RandomMat(qdim, src_seqlen, -1.f, 1.f);
        as[1] = RandomMat(kdim, 2, -1.f, 1.f);
        as[2] = RandomMat(vdim, 2, -1.f, 1.f);
    }
    if (attn_mask)
    {
        as[3] = RandomMat(5, src_seqlen, -1.f, 0.f);
        as[4] = RandomMat(5, embed_dim, -1.f, 1.f);
        as[5] = bits == 8 && has_input_scale ? RandomWQInt8Cache(5, embed_dim) : RandomMat(5, embed_dim, -1.f, 1.f);
    }
    else
    {
        as[3] = RandomMat(5, embed_dim, -1.f, 1.f);
        as[4] = bits == 8 && has_input_scale ? RandomWQInt8Cache(5, embed_dim) : RandomMat(5, embed_dim, -1.f, 1.f);
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

    const int ret = test_multiheadattention_block_quant(pd, weights, ref_weights, as, 3, bits);

    if (ret != 0)
        fprintf(stderr, "test_multiheadattention_block_quant_cross_kvcache failed bits=%d block=%d mask=%d input_scale=%d\n", bits, block_size, attn_mask, has_input_scale);

    return ret;
}

static int test_multiheadattention_wq_int8_pipeline()
{
    const int qdim = 35;
    const int embed_dim = 32;
    const int block_size = 32;

    std::vector<ncnn::Mat> weights;
    std::vector<ncnn::Mat> ref_weights;
    make_mha_weights(qdim, qdim, qdim, embed_dim, 8, block_size, 1, weights, ref_weights);

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

    mha->load_param(pd);
    mha->load_model(ncnn::ModelBinFromMatArray(weights.data()));
    mha->create_pipeline(opt);

    std::vector<ncnn::Mat> prefill_inputs(3);
    prefill_inputs[0] = RandomWQInt8Mat(qdim, 9, block_size, weights[12]);

    std::vector<ncnn::Mat> prefill_reference;
    std::vector<ncnn::Mat> prefill_outputs(3);
    test_layer_naive(ncnn::layer_to_index("MultiHeadAttention"), pd, weights, prefill_inputs, 3, prefill_reference, TEST_LAYER_DISABLE_GPU_TESTING);
    mha->forward(prefill_inputs, prefill_outputs, opt);

    int test_ret = CompareMat(prefill_outputs, prefill_reference, 0.001f);

    for (int i = 0; i < 3; i++)
    {
        if (prefill_outputs[i].elembits() != 32 || prefill_outputs[i].elempack != 1)
            test_ret = -1;
    }
    if (prefill_outputs[1].w != 9 || prefill_outputs[2].w != 9)
        test_ret = -1;

    std::vector<ncnn::Mat> decode_reference_inputs(3);
    std::vector<ncnn::Mat> decode_inputs(3);
    decode_inputs[0] = RandomWQInt8Mat(qdim, 1, block_size, weights[12]);
    decode_reference_inputs[0] = decode_inputs[0];
    decode_reference_inputs[1] = prefill_reference[1];
    decode_reference_inputs[2] = prefill_reference[2];
    decode_inputs[1] = prefill_outputs[1];
    decode_inputs[2] = prefill_outputs[2];

    std::vector<ncnn::Mat> decode_reference;
    std::vector<ncnn::Mat> decode_outputs(3);
    test_layer_naive(ncnn::layer_to_index("MultiHeadAttention"), pd, weights, decode_reference_inputs, 3, decode_reference, TEST_LAYER_DISABLE_GPU_TESTING);
    ncnn::Option decode_opt = opt;
    decode_opt.num_threads = 4;
    mha->forward(decode_inputs, decode_outputs, decode_opt);
    if (CompareMat(decode_outputs, decode_reference, 0.001f) != 0)
        test_ret = -1;

    for (int i = 0; i < 3; i++)
    {
        if (decode_outputs[i].elembits() != 32 || decode_outputs[i].elempack != 1)
            test_ret = -1;
    }
    if (decode_outputs[1].w != 10 || decode_outputs[2].w != 10)
        test_ret = -1;

    mha->destroy_pipeline(opt);
    delete mha;

    if (test_ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_wq_int8_pipeline failed ret=%d\n", test_ret);
        return test_ret;
    }

    return 0;
}

static int test_multiheadattention_block_quant_0()
{
    return 0
           || test_multiheadattention_block_quant(13, 9, 11, 8, 2, 4, 32, 0, 0)
           || test_multiheadattention_block_quant(10, 10, 10, 8, 2, 6, 64, 1, 0)
           || test_multiheadattention_block_quant(12, 7, 9, 8, 2, 8, 128, 0, 0)
           || test_multiheadattention_block_quant(35, 33, 31, 32, 4, 8, 32, 1, 0)
           || test_multiheadattention_block_quant(65, 33, 49, 64, 4, 8, 64, 0, 1)
           || test_multiheadattention_block_quant(129, 129, 129, 128, 8, 8, 128, 1, 1, 1);
}

static int test_multiheadattention_block_quant_1()
{
    return 0
           || test_multiheadattention_block_quant_kvcache(4, 64, 0, 0)
           || test_multiheadattention_block_quant_kvcache(8, 32, 0, 0)
           || test_multiheadattention_block_quant_kvcache(8, 64, 1, 1);
}

static int test_multiheadattention_block_quant_2()
{
    return 0
           || test_multiheadattention_block_quant_cross_kvcache(6, 32, 0, 0)
           || test_multiheadattention_block_quant_cross_kvcache(8, 32, 0, 0)
           || test_multiheadattention_block_quant_cross_kvcache(8, 128, 1, 1);
}

#endif // NCNN_WEIGHT_QUANT

int main()
{
    SRAND(7767517);

#if NCNN_WEIGHT_QUANT
    return 0
           || test_multiheadattention_block_quant_0()
           || test_multiheadattention_block_quant_1()
           || test_multiheadattention_block_quant_2()
           || test_multiheadattention_wq_int8_pipeline();
#else
    return 0;
#endif
}
