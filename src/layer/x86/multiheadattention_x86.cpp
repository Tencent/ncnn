// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "multiheadattention_x86.h"

#include "layer_type.h"

namespace ncnn {

MultiHeadAttention_x86::MultiHeadAttention_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__

#if NCNN_BF16
    support_bf16_storage = true;
#endif

    q_gemm = 0;
    k_gemm = 0;
    v_gemm = 0;

    qk_gemm = 0;
    qkv_gemm = 0;

    qk_softmax = 0;

    o_gemm = 0;
}

#if NCNN_WEIGHT_QUANT
int MultiHeadAttention_x86::create_pipeline_wq_int8(const Option& _opt)
{
    if (q_gemm)
        return 0;

    Option opt = _opt;
    Option opt_wq = opt;
    opt_wq.use_packing_layout = false;
    opt_wq.use_fp16_packed = false;
    opt_wq.use_fp16_storage = false;
    opt_wq.use_fp16_arithmetic = false;
    opt_wq.use_bf16_packed = false;
    opt_wq.use_bf16_storage = false;

    {
        qk_softmax = ncnn::create_layer_cpu(ncnn::LayerType::Softmax);
        if (!qk_softmax)
            return -100;
        ncnn::ParamDict pd;
        pd.set(0, -1);
        pd.set(1, 1);
        int ret = qk_softmax->load_param(pd);
        if (ret != 0)
        {
            destroy_pipeline(_opt);
            return ret;
        }
        ret = qk_softmax->load_model(ModelBinFromMatArray(0));
        if (ret != 0)
        {
            destroy_pipeline(_opt);
            return ret;
        }
        ret = qk_softmax->create_pipeline(opt);
        if (ret != 0)
        {
            destroy_pipeline(_opt);
            return ret;
        }
    }

    const int qdim = weight_data_size / embed_dim;

    {
        q_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        if (!q_gemm)
        {
            destroy_pipeline(_opt);
            return -100;
        }
        ncnn::ParamDict pd;
        pd.set(0, scale);
        pd.set(1, 1.f);
        pd.set(2, 0);         // transA
        pd.set(3, 1);         // transB
        pd.set(4, 0);         // constantA
        pd.set(5, 1);         // constantB
        pd.set(6, 1);         // constantC
        pd.set(7, 0);         // M
        pd.set(8, embed_dim); // N
        pd.set(9, qdim);      // K
        pd.set(10, 4);        // constant_broadcast_type_C
        pd.set(11, 0);        // output_N1M
        pd.set(12, 0);        // output_elempack
        pd.set(14, 1);        // output_transpose
        pd.set(18, quantize_term);
        int ret = q_gemm->load_param(pd);
        if (ret != 0)
        {
            destroy_pipeline(_opt);
            return ret;
        }
        Mat weights[4];
        weights[0] = q_weight_data;
        weights[1] = q_bias_data;
        weights[2] = q_weight_data_quantize_scales;
        weights[3] = q_weight_data_input_scales;
        ret = q_gemm->load_model(ModelBinFromMatArray(weights));
        if (ret != 0)
        {
            destroy_pipeline(_opt);
            return ret;
        }
        ret = q_gemm->create_pipeline(opt_wq);
        if (ret != 0)
        {
            destroy_pipeline(_opt);
            return ret;
        }
    }

    {
        k_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        if (!k_gemm)
        {
            destroy_pipeline(_opt);
            return -100;
        }
        ncnn::ParamDict pd;
        pd.set(2, 0);         // transA
        pd.set(3, 1);         // transB
        pd.set(4, 0);         // constantA
        pd.set(5, 1);         // constantB
        pd.set(6, 1);         // constantC
        pd.set(7, 0);         // M
        pd.set(8, embed_dim); // N
        pd.set(9, kdim);      // K
        pd.set(10, 4);        // constant_broadcast_type_C
        pd.set(11, 0);        // output_N1M
        pd.set(12, 0);        // output_elempack
        pd.set(14, 1);        // output_transpose
        pd.set(18, quantize_term);
        int ret = k_gemm->load_param(pd);
        if (ret != 0)
        {
            destroy_pipeline(_opt);
            return ret;
        }
        Mat weights[4];
        weights[0] = k_weight_data;
        weights[1] = k_bias_data;
        weights[2] = k_weight_data_quantize_scales;
        weights[3] = k_weight_data_input_scales;
        ret = k_gemm->load_model(ModelBinFromMatArray(weights));
        if (ret != 0)
        {
            destroy_pipeline(_opt);
            return ret;
        }
        ret = k_gemm->create_pipeline(opt_wq);
        if (ret != 0)
        {
            destroy_pipeline(_opt);
            return ret;
        }
    }

    {
        v_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        if (!v_gemm)
        {
            destroy_pipeline(_opt);
            return -100;
        }
        ncnn::ParamDict pd;
        pd.set(2, 0);         // transA
        pd.set(3, 1);         // transB
        pd.set(4, 0);         // constantA
        pd.set(5, 1);         // constantB
        pd.set(6, 1);         // constantC
        pd.set(7, 0);         // M
        pd.set(8, embed_dim); // N
        pd.set(9, vdim);      // K
        pd.set(10, 4);        // constant_broadcast_type_C
        pd.set(11, 0);        // output_N1M
        pd.set(12, 0);        // output_elempack
        pd.set(14, 1);        // output_transpose
        pd.set(18, quantize_term);
        int ret = v_gemm->load_param(pd);
        if (ret != 0)
        {
            destroy_pipeline(_opt);
            return ret;
        }
        Mat weights[4];
        weights[0] = v_weight_data;
        weights[1] = v_bias_data;
        weights[2] = v_weight_data_quantize_scales;
        weights[3] = v_weight_data_input_scales;
        ret = v_gemm->load_model(ModelBinFromMatArray(weights));
        if (ret != 0)
        {
            destroy_pipeline(_opt);
            return ret;
        }
        ret = v_gemm->create_pipeline(opt_wq);
        if (ret != 0)
        {
            destroy_pipeline(_opt);
            return ret;
        }
    }

    {
        o_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        if (!o_gemm)
        {
            destroy_pipeline(_opt);
            return -100;
        }
        ncnn::ParamDict pd;
        pd.set(2, 1);         // transA
        pd.set(3, 1);         // transB
        pd.set(4, 0);         // constantA
        pd.set(5, 1);         // constantB
        pd.set(6, 1);         // constantC
        pd.set(7, 0);         // M = outch
        pd.set(8, qdim);      // N = size
        pd.set(9, embed_dim); // K = maxk*inch
        pd.set(10, 4);        // constant_broadcast_type_C
        pd.set(11, 0);        // output_N1M
        pd.set(18, quantize_term);
        int ret = o_gemm->load_param(pd);
        if (ret != 0)
        {
            destroy_pipeline(_opt);
            return ret;
        }
        Mat weights[4];
        weights[0] = out_weight_data;
        weights[1] = out_bias_data;
        weights[2] = out_weight_data_quantize_scales;
        weights[3] = out_weight_data_input_scales;
        ret = o_gemm->load_model(ModelBinFromMatArray(weights));
        if (ret != 0)
        {
            destroy_pipeline(_opt);
            return ret;
        }
        ret = o_gemm->create_pipeline(opt_wq);
        if (ret != 0)
        {
            destroy_pipeline(_opt);
            return ret;
        }
    }

    {
        qk_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        if (!qk_gemm)
        {
            destroy_pipeline(_opt);
            return -100;
        }
        ncnn::ParamDict pd;
        pd.set(2, 1);                   // transA
        pd.set(3, kv_cache);            // transB
        pd.set(4, 0);                   // constantA
        pd.set(5, 0);                   // constantB
        pd.set(6, attn_mask ? 0 : 1);   // constantC
        pd.set(7, 0);                   // M
        pd.set(8, 0);                   // N
        pd.set(9, 0);                   // K
        pd.set(10, attn_mask ? 3 : -1); // constant_broadcast_type_C
        pd.set(11, 0);                  // output_N1M
        pd.set(12, 1);                  // output_elempack
        pd.set(13, 1);                  // output_elemtype = fp32
        int ret = qk_gemm->load_param(pd);
        if (ret != 0)
        {
            destroy_pipeline(_opt);
            return ret;
        }
        ret = qk_gemm->load_model(ModelBinFromMatArray(0));
        if (ret != 0)
        {
            destroy_pipeline(_opt);
            return ret;
        }
        Option opt1 = opt;
        opt1.use_bf16_packed = false;
        opt1.use_bf16_storage = false;
        opt1.num_threads = 1;
        ret = qk_gemm->create_pipeline(opt1);
        if (ret != 0)
        {
            destroy_pipeline(_opt);
            return ret;
        }
    }

    {
        qkv_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        if (!qkv_gemm)
        {
            destroy_pipeline(_opt);
            return -100;
        }
        ncnn::ParamDict pd;
        pd.set(2, 0);         // transA
        pd.set(3, !kv_cache); // transB
        pd.set(4, 0);         // constantA
        pd.set(5, 0);         // constantB
        pd.set(6, 1);         // constantC
        pd.set(7, 0);         // M
        pd.set(8, 0);         // N
        pd.set(9, 0);         // K
        pd.set(10, -1);       // constant_broadcast_type_C
        pd.set(11, 0);        // output_N1M
        pd.set(12, 1);        // output_elempack
        pd.set(13, 1);        // output_elemtype = fp32
        pd.set(14, 1);        // output_transpose
        int ret = qkv_gemm->load_param(pd);
        if (ret != 0)
        {
            destroy_pipeline(_opt);
            return ret;
        }
        ret = qkv_gemm->load_model(ModelBinFromMatArray(0));
        if (ret != 0)
        {
            destroy_pipeline(_opt);
            return ret;
        }
        Option opt1 = opt;
        opt1.use_bf16_packed = false;
        opt1.use_bf16_storage = false;
        opt1.num_threads = 1;
        ret = qkv_gemm->create_pipeline(opt1);
        if (ret != 0)
        {
            destroy_pipeline(_opt);
            return ret;
        }
    }

    if (_opt.lightmode)
    {
        q_weight_data.release();
        q_bias_data.release();
        k_weight_data.release();
        k_bias_data.release();
        v_weight_data.release();
        v_bias_data.release();
        out_weight_data.release();
        out_bias_data.release();
        q_weight_data_quantize_scales.release();
        k_weight_data_quantize_scales.release();
        v_weight_data_quantize_scales.release();
        out_weight_data_quantize_scales.release();
        q_weight_data_input_scales.release();
        k_weight_data_input_scales.release();
        v_weight_data_input_scales.release();
        out_weight_data_input_scales.release();
    }

    return 0;
}
#endif // NCNN_WEIGHT_QUANT

int MultiHeadAttention_x86::create_pipeline(const Option& _opt)
{
#if NCNN_WEIGHT_QUANT
    if (weight_block_quantize)
    {
        int weight_bits;
        int block_size;
        bool has_input_scale;
        const int ret = get_weight_block_quantize_params(weight_bits, block_size, has_input_scale);
        if (ret != 0)
            return ret;

        if (weight_bits != 8)
            return MultiHeadAttention::create_pipeline(_opt);

        return create_pipeline_wq_int8(_opt);
    }
#endif

    Option opt = _opt;
    if (int8_scale_term)
    {
        support_packing = false;
        support_bf16_storage = false;

        opt.use_packing_layout = false; // TODO enable packing
    }

    {
        qk_softmax = ncnn::create_layer_cpu(ncnn::LayerType::Softmax);
        if (!qk_softmax)
            return -100;
        ncnn::ParamDict pd;
        pd.set(0, -1);
        pd.set(1, 1);
        int ret = qk_softmax->load_param(pd);
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
        ret = qk_softmax->load_model(ModelBinFromMatArray(0));
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
        ret = qk_softmax->create_pipeline(opt);
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
    }

    const int qdim = weight_data_size / embed_dim;

    {
        q_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        if (!q_gemm)
        {
            destroy_pipeline(opt);
            return -100;
        }
        ncnn::ParamDict pd;
        pd.set(0, scale);
        pd.set(1, 1.f);
        pd.set(2, 0);         // transA
        pd.set(3, 1);         // transB
        pd.set(4, 1);         // constantA
        pd.set(5, 0);         // constantB
        pd.set(6, 1);         // constantC
        pd.set(7, embed_dim); // M
        pd.set(8, 0);         // N
        pd.set(9, qdim);      // K
        pd.set(10, 1);        // constant_broadcast_type_C
        pd.set(11, 0);        // output_N1M
        pd.set(12, 1);        // output_elempack
        pd.set(14, 0);        // output_transpose
#if NCNN_INT8
        pd.set(18, int8_scale_term);
#endif
        int ret = q_gemm->load_param(pd);
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
        Mat weights[3];
        weights[0] = q_weight_data;
        weights[1] = q_bias_data;
#if NCNN_INT8
        weights[2] = q_weight_data_int8_scales;
#endif
        ret = q_gemm->load_model(ModelBinFromMatArray(weights));
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
        ret = q_gemm->create_pipeline(opt);
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
    }

    {
        k_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        if (!k_gemm)
        {
            destroy_pipeline(opt);
            return -100;
        }
        ncnn::ParamDict pd;
        pd.set(2, 0);         // transA
        pd.set(3, 1);         // transB
        pd.set(4, 1);         // constantA
        pd.set(5, 0);         // constantB
        pd.set(6, 1);         // constantC
        pd.set(7, embed_dim); // M
        pd.set(8, 0);         // N
        pd.set(9, kdim);      // K
        pd.set(10, 1);        // constant_broadcast_type_C
        pd.set(11, 0);        // output_N1M
        pd.set(12, 1);        // output_elempack
        pd.set(14, 0);        // output_transpose
#if NCNN_INT8
        pd.set(18, int8_scale_term);
#endif
        int ret = k_gemm->load_param(pd);
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
        Mat weights[3];
        weights[0] = k_weight_data;
        weights[1] = k_bias_data;
#if NCNN_INT8
        weights[2] = k_weight_data_int8_scales;
#endif
        ret = k_gemm->load_model(ModelBinFromMatArray(weights));
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
        ret = k_gemm->create_pipeline(opt);
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
    }

    {
        v_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        if (!v_gemm)
        {
            destroy_pipeline(opt);
            return -100;
        }
        ncnn::ParamDict pd;
        pd.set(2, 0);         // transA
        pd.set(3, 1);         // transB
        pd.set(4, 1);         // constantA
        pd.set(5, 0);         // constantB
        pd.set(6, 1);         // constantC
        pd.set(7, embed_dim); // M
        pd.set(8, 0);         // N
        pd.set(9, vdim);      // K
        pd.set(10, 1);        // constant_broadcast_type_C
        pd.set(11, 0);        // output_N1M
        pd.set(12, 1);        // output_elempack
        pd.set(14, 0);        // output_transpose
#if NCNN_INT8
        pd.set(18, int8_scale_term);
#endif
        int ret = v_gemm->load_param(pd);
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
        Mat weights[3];
        weights[0] = v_weight_data;
        weights[1] = v_bias_data;
#if NCNN_INT8
        weights[2] = v_weight_data_int8_scales;
#endif
        ret = v_gemm->load_model(ModelBinFromMatArray(weights));
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
        ret = v_gemm->create_pipeline(opt);
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
    }

    {
        o_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        if (!o_gemm)
        {
            destroy_pipeline(opt);
            return -100;
        }
        ncnn::ParamDict pd;
        pd.set(2, 1);         // transA
        pd.set(3, 1);         // transB
        pd.set(4, 0);         // constantA
        pd.set(5, 1);         // constantB
        pd.set(6, 1);         // constantC
        pd.set(7, 0);         // M = outch
        pd.set(8, qdim);      // N = size
        pd.set(9, embed_dim); // K = maxk*inch
        pd.set(10, 4);        // constant_broadcast_type_C
        pd.set(11, 0);        // output_N1M
#if NCNN_INT8
        pd.set(18, int8_scale_term);
#endif
        int ret = o_gemm->load_param(pd);
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
        Mat weights[3];
        weights[0] = out_weight_data;
        weights[1] = out_bias_data;
#if NCNN_INT8
        Mat out_weight_data_int8_scales(1);
        if (out_weight_data_int8_scales.empty())
        {
            destroy_pipeline(opt);
            return -100;
        }
        out_weight_data_int8_scales[0] = out_weight_data_int8_scale;
        weights[2] = out_weight_data_int8_scales;
#endif
        ret = o_gemm->load_model(ModelBinFromMatArray(weights));
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
        Option opt_fp32 = opt;
        opt_fp32.use_bf16_packed = false;
        opt_fp32.use_bf16_storage = false;
        ret = o_gemm->create_pipeline(opt_fp32);
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
    }

    {
        qk_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        if (!qk_gemm)
        {
            destroy_pipeline(opt);
            return -100;
        }
        ncnn::ParamDict pd;
        pd.set(2, 1);                   // transA
        pd.set(3, kv_cache);            // transB
        pd.set(4, 0);                   // constantA
        pd.set(5, 0);                   // constantB
        pd.set(6, attn_mask ? 0 : 1);   // constantC
        pd.set(7, 0);                   // M
        pd.set(8, 0);                   // N
        pd.set(9, 0);                   // K
        pd.set(10, attn_mask ? 3 : -1); // constant_broadcast_type_C
        pd.set(11, 0);                  // output_N1M
        pd.set(12, 1);                  // output_elempack
        pd.set(13, 1);                  // output_elemtype = fp32
#if NCNN_INT8
        pd.set(18, int8_scale_term);
#endif
        int ret = qk_gemm->load_param(pd);
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
        ret = qk_gemm->load_model(ModelBinFromMatArray(0));
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
        Option opt1 = opt;
        opt1.use_bf16_packed = false;
        opt1.use_bf16_storage = false;
        opt1.num_threads = 1;
        ret = qk_gemm->create_pipeline(opt1);
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
    }

    {
        qkv_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        if (!qkv_gemm)
        {
            destroy_pipeline(opt);
            return -100;
        }
        ncnn::ParamDict pd;
        pd.set(2, 0);         // transA
        pd.set(3, !kv_cache); // transB
        pd.set(4, 0);         // constantA
        pd.set(5, 0);         // constantB
        pd.set(6, 1);         // constantC
        pd.set(7, 0);         // M
        pd.set(8, 0);         // N
        pd.set(9, 0);         // K
        pd.set(10, -1);       // constant_broadcast_type_C
        pd.set(11, 0);        // output_N1M
        pd.set(12, 1);        // output_elempack
        pd.set(13, 1);        // output_elemtype = fp32
        pd.set(14, 1);        // output_transpose
#if NCNN_INT8
        pd.set(18, int8_scale_term);
#endif
        int ret = qkv_gemm->load_param(pd);
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
        ret = qkv_gemm->load_model(ModelBinFromMatArray(0));
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
        Option opt1 = opt;
        opt1.use_bf16_packed = false;
        opt1.use_bf16_storage = false;
        opt1.num_threads = 1;
        ret = qkv_gemm->create_pipeline(opt1);
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
    }

    if (opt.lightmode)
    {
        q_weight_data.release();
        q_bias_data.release();
        k_weight_data.release();
        k_bias_data.release();
        v_weight_data.release();
        v_bias_data.release();
        out_weight_data.release();
        out_bias_data.release();
    }

    return 0;
}

int MultiHeadAttention_x86::destroy_pipeline(const Option& _opt)
{
    if (weight_block_quantize)
    {
        int weight_bits;
        int block_size;
        bool has_input_scale;
        const int ret = get_weight_block_quantize_params(weight_bits, block_size, has_input_scale);
        if (ret != 0)
            return ret;

        if (weight_bits != 8)
            return MultiHeadAttention::destroy_pipeline(_opt);
    }

    Option opt = _opt;
    if (int8_scale_term && !weight_block_quantize)
    {
        opt.use_packing_layout = false; // TODO enable packing
    }
    Option opt_wq = opt;
    if (weight_block_quantize)
    {
        opt_wq.use_packing_layout = false;
        opt_wq.use_fp16_packed = false;
        opt_wq.use_fp16_storage = false;
        opt_wq.use_fp16_arithmetic = false;
        opt_wq.use_bf16_packed = false;
        opt_wq.use_bf16_storage = false;
    }

    if (qk_softmax)
    {
        qk_softmax->destroy_pipeline(opt);
        delete qk_softmax;
        qk_softmax = 0;
    }

    if (q_gemm)
    {
        q_gemm->destroy_pipeline(opt_wq);
        delete q_gemm;
        q_gemm = 0;
    }

    if (k_gemm)
    {
        k_gemm->destroy_pipeline(opt_wq);
        delete k_gemm;
        k_gemm = 0;
    }

    if (v_gemm)
    {
        v_gemm->destroy_pipeline(opt_wq);
        delete v_gemm;
        v_gemm = 0;
    }

    if (o_gemm)
    {
        o_gemm->destroy_pipeline(opt_wq);
        delete o_gemm;
        o_gemm = 0;
    }

    if (qk_gemm)
    {
        qk_gemm->destroy_pipeline(opt);
        delete qk_gemm;
        qk_gemm = 0;
    }
    if (qkv_gemm)
    {
        qkv_gemm->destroy_pipeline(opt);
        delete qkv_gemm;
        qkv_gemm = 0;
    }

    return 0;
}

int MultiHeadAttention_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& _opt) const
{
#if NCNN_WEIGHT_QUANT
    if (weight_block_quantize)
    {
        int weight_bits;
        int block_size;
        bool has_input_scale;
        const int ret = get_weight_block_quantize_params(weight_bits, block_size, has_input_scale);
        if (ret != 0)
            return ret;

        if (weight_bits != 8)
            return MultiHeadAttention::forward(bottom_blobs, top_blobs, _opt);
    }
#endif

    int q_blob_i = 0;
    int k_blob_i = 0;
    int v_blob_i = 0;
    int attn_mask_i = 0;
    int cached_xk_i = 0;
    int cached_xv_i = 0;
    resolve_bottom_blob_index((int)bottom_blobs.size(), q_blob_i, k_blob_i, v_blob_i, attn_mask_i, cached_xk_i, cached_xv_i);

    const Mat& q_blob = bottom_blobs[q_blob_i];
    const Mat& k_blob = bottom_blobs[k_blob_i];
    const Mat& v_blob = bottom_blobs[v_blob_i];
    const Mat& attn_mask_blob = attn_mask ? bottom_blobs[attn_mask_i] : Mat();
    Mat empty_cache;
    const Mat& past_xk_blob = kv_cache ? bottom_blobs[cached_xk_i] : empty_cache;
    const Mat& past_xv_blob = kv_cache ? bottom_blobs[cached_xv_i] : empty_cache;
    Mat& cached_xk_blob = kv_cache ? top_blobs[1] : empty_cache;
    Mat& cached_xv_blob = kv_cache ? top_blobs[2] : empty_cache;

    Option opt = _opt;
    if (int8_scale_term && !weight_block_quantize)
    {
        opt.use_packing_layout = false; // TODO enable packing
    }
    Option opt_wq = opt;
#if NCNN_WEIGHT_QUANT
    if (weight_block_quantize)
    {
        opt_wq.use_packing_layout = false;
        opt_wq.use_fp16_packed = false;
        opt_wq.use_fp16_storage = false;
        opt_wq.use_fp16_arithmetic = false;
        opt_wq.use_bf16_packed = false;
        opt_wq.use_bf16_storage = false;
    }
#endif

    Mat attn_mask_blob_unpacked;
    if (attn_mask && attn_mask_blob.elempack != 1)
    {
        convert_packing(attn_mask_blob, attn_mask_blob_unpacked, 1, opt);
        if (attn_mask_blob_unpacked.empty())
            return -100;
    }
    else
    {
        attn_mask_blob_unpacked = attn_mask_blob;
    }

    Mat past_xk_blob_unpacked;
    if (kv_cache && !past_xk_blob.empty() && past_xk_blob.elempack != 1)
    {
        convert_packing(past_xk_blob, past_xk_blob_unpacked, 1, opt);
        if (past_xk_blob_unpacked.empty())
            return -100;
    }
    else
    {
        past_xk_blob_unpacked = past_xk_blob;
    }

    Mat past_xv_blob_unpacked;
    if (kv_cache && !past_xv_blob.empty() && past_xv_blob.elempack != 1)
    {
        convert_packing(past_xv_blob, past_xv_blob_unpacked, 1, opt);
        if (past_xv_blob_unpacked.empty())
            return -100;
    }
    else
    {
        past_xv_blob_unpacked = past_xv_blob;
    }

    const int embed_dim_per_head = embed_dim / num_heads;
    const int src_seqlen = q_blob.h * q_blob.elempack;
    const int cur_seqlen = k_blob.h * k_blob.elempack;
    const int past_seqlen = kv_cache && !past_xk_blob_unpacked.empty() ? past_xk_blob_unpacked.h : 0;
    const int dst_seqlen = past_seqlen > 0 ? (q_blob_i == k_blob_i ? (past_seqlen + cur_seqlen) : past_seqlen) : cur_seqlen;

    Mat q_affine;
    int retq = q_gemm->forward(q_blob, q_affine, opt_wq);
    if (retq != 0)
        return retq;

    Mat k_affine;
    if (kv_cache)
    {
        const bool append_kv = past_seqlen == 0 || q_blob_i == k_blob_i;
        const int append_seqlen = append_kv ? cur_seqlen : 0;
        Mat current_key;
        Mat current_value;
        if (append_seqlen > 0)
        {
            int retk = k_gemm->forward(k_blob, current_key, opt_wq);
            if (retk != 0)
                return retk;

            int retv = v_gemm->forward(v_blob, current_value, opt_wq);
            if (retv != 0)
                return retv;
        }
        int retk = create_or_grow_kvcache(past_xk_blob_unpacked, cached_xk_blob, dst_seqlen, num_heads, embed_dim_per_head, current_key.elemsize, 1, opt);
        if (retk != 0)
            return retk;

        int retv = create_or_grow_kvcache(past_xv_blob_unpacked, cached_xv_blob, dst_seqlen, num_heads, embed_dim_per_head, current_value.elemsize, 1, opt);
        if (retv != 0)
            return retv;

        if (append_seqlen > 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < num_heads; q++)
            {
                Mat key_cache_head = cached_xk_blob.channel(q);
                Mat value_cache_head = cached_xv_blob.channel(q);

                unsigned char* key_outptr = key_cache_head.row<unsigned char>(past_seqlen);
                unsigned char* value_outptr = value_cache_head.row<unsigned char>(past_seqlen);
                for (int d = 0; d < embed_dim_per_head; d++)
                {
                    const unsigned char* key_ptr = current_key.row<const unsigned char>(q * embed_dim_per_head + d);
                    const unsigned char* value_ptr = current_value.row<const unsigned char>(q * embed_dim_per_head + d);
                    for (int s = 0; s < append_seqlen; s++)
                    {
                        memcpy(key_outptr + ((size_t)s * embed_dim_per_head + d) * cached_xk_blob.elemsize, key_ptr + (size_t)s * cached_xk_blob.elemsize, cached_xk_blob.elemsize);
                        memcpy(value_outptr + ((size_t)s * embed_dim_per_head + d) * cached_xv_blob.elemsize, value_ptr + (size_t)s * cached_xv_blob.elemsize, cached_xv_blob.elemsize);
                    }
                }
            }
        }
    }
    else
    {
        int retk = k_gemm->forward(k_blob, k_affine, opt_wq);
        if (retk != 0)
            return retk;
    }

    Mat qk_cross(dst_seqlen, src_seqlen * num_heads, 4u, opt.blob_allocator);
    if (qk_cross.empty())
        return -100;

    std::vector<int> retqks;
    retqks.resize(num_heads);
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < num_heads; i++)
    {
        std::vector<Mat> qk_bottom_blobs(2);
        qk_bottom_blobs[0] = q_affine.row_range(i * embed_dim_per_head, embed_dim_per_head);
        qk_bottom_blobs[1] = kv_cache ? cached_xk_blob.channel(i) : k_affine.row_range(i * embed_dim_per_head, embed_dim_per_head);
        if (attn_mask)
        {
            const Mat& maskm = attn_mask_blob_unpacked.dims == 3 ? attn_mask_blob_unpacked.channel(i) : attn_mask_blob_unpacked;
            qk_bottom_blobs.push_back(maskm);
        }
        std::vector<Mat> qk_top_blobs(1);
        qk_top_blobs[0] = qk_cross.row_range(i * src_seqlen, src_seqlen);
        Option opt1 = opt;
        opt1.num_threads = 1;
        retqks[i] = qk_gemm->forward(qk_bottom_blobs, qk_top_blobs, opt1);
    }
    for (int i = 0; i < num_heads; i++)
    {
        if (retqks[i] != 0)
            return retqks[i];
    }

    q_affine.release();

    if (!kv_cache)
    {
        k_affine.release();
    }

    int retqk = qk_softmax->forward_inplace(qk_cross, opt);
    if (retqk != 0)
        return retqk;

    Mat v_affine;
    if (!kv_cache)
    {
        int retv = v_gemm->forward(v_blob, v_affine, opt_wq);
        if (retv != 0)
            return retv;
    }

    const Mat& value_affine = kv_cache ? cached_xv_blob : v_affine;
    Mat v_affine_fp32 = value_affine;

#if NCNN_BF16
    if (opt.use_bf16_storage && value_affine.elembits() == 16)
    {
        // qkv_gemm need fp32 inputs
        cast_bfloat16_to_float32(value_affine, v_affine_fp32, opt_wq);
        if (v_affine_fp32.empty())
            return -100;
    }
#endif

    Mat qkv_cross(src_seqlen, embed_dim_per_head * num_heads, 4u, opt.blob_allocator);
    if (qkv_cross.empty())
        return -100;

    std::vector<int> retqkvs;
    retqkvs.resize(num_heads);
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < num_heads; i++)
    {
        std::vector<Mat> qkv_bottom_blobs(2);
        qkv_bottom_blobs[0] = qk_cross.row_range(i * src_seqlen, src_seqlen);
        qkv_bottom_blobs[1] = kv_cache ? v_affine_fp32.channel(i) : v_affine_fp32.row_range(i * embed_dim_per_head, embed_dim_per_head);
        std::vector<Mat> qkv_top_blobs(1);
        qkv_top_blobs[0] = qkv_cross.row_range(i * embed_dim_per_head, embed_dim_per_head);
        Option opt1 = opt;
        opt1.num_threads = 1;
        retqkvs[i] = qkv_gemm->forward(qkv_bottom_blobs, qkv_top_blobs, opt1);
    }
    for (int i = 0; i < num_heads; i++)
    {
        if (retqkvs[i] != 0)
            return retqkvs[i];
    }

    v_affine_fp32.release();

    if (!kv_cache)
    {
        v_affine.release();
    }

    int reto = o_gemm->forward(qkv_cross, top_blobs[0], opt_wq);
    if (reto != 0)
        return reto;

    return 0;
}

} // namespace ncnn
