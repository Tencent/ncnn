// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#if NCNN_INT8 && (defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64))
static int test_convolution_int8_flattened(bool packing, bool input_int8, bool requant, int pad)
{
    const int inch = 64;
    const int outch = 16;

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = packing;
    opt.use_bf16_storage = false;
    opt.use_fp16_storage = false;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, 1);
    pd.set(4, pad);
    pd.set(5, 1);
    pd.set(6, inch * outch);
    pd.set(8, requant ? 102 : 2);
    pd.set(9, 1);

    ncnn::Mat weights[5];
    weights[0] = RandomMat(inch * outch);
    weights[1] = RandomMat(outch);
    weights[2] = ncnn::Mat(outch);
    weights[2].fill(8.f);
    weights[3] = ncnn::Mat(1);
    weights[3].fill(8.f);
    weights[4] = ncnn::Mat(1);
    weights[4].fill(8.f);

    ncnn::Layer* op = ncnn::create_layer_cpu(ncnn::LayerType::Convolution);
    if (packing && !op->support_packing)
    {
        delete op;
        return 0;
    }

    ncnn::Mat a = RandomMat(inch);
    if (input_int8)
    {
        ncnn::Option opt_q = opt;
        opt_q.use_packing_layout = false;
        ncnn::Mat a_int8;
        ncnn::quantize_to_int8(a, a_int8, weights[3], opt_q);
        a = a_int8;
    }
    int input_pack = 1;
    if (packing)
    {
        input_pack = input_int8 ? 8 : 4;
        // Match the FP32 packing selected by Net for the x86 quantizer.
#if NCNN_AVX || NCNN_AVX512
        if (!input_int8 && ncnn::cpu_support_x86_avx())
            input_pack = 8;
#endif
#if NCNN_AVX512
        if (!input_int8 && ncnn::cpu_support_x86_avx512())
            input_pack = 16;
#endif
        ncnn::Mat a_packed;
        ncnn::convert_packing(a, a_packed, input_pack, opt);
        a = a_packed;
    }
    ncnn::Mat a_spatial = a.reshape(1, 1, a.w);
    ncnn::Mat out;
    ncnn::Mat out_spatial;

    int ret = op->load_param(pd);
    if (ret == 0)
        ret = op->load_model(ncnn::ModelBinFromMatArray(weights));
    if (ret == 0)
        ret = op->create_pipeline(opt);
    if (ret == 0)
        ret = op->forward(a_spatial, out_spatial, opt);
    const int flat_ret = ret == 0 ? op->forward(a, out, opt) : ret;
    op->destroy_pipeline(opt);
    delete op;

    const int spatial_size = pad > 0 ? 1 + 2 * pad : 1;
    const bool spatial_shape = out_spatial.dims == 3 && out_spatial.w == spatial_size && out_spatial.h == spatial_size && out_spatial.c * out_spatial.elempack == outch;
    bool flat_ok = flat_ret == -1 && out.empty();
    if (pad <= 0)
    {
        const bool flat_shape = out.dims == 1 && out.w * out.elempack == outch && out.elembits() == (requant ? 8 : 32) && out.elempack == out_spatial.elempack;
        flat_ok = flat_ret == 0 && flat_shape && spatial_shape && CompareMat(out, out_spatial.reshape(out_spatial.c), 0.0001f) == 0;
    }
    if (ret != 0 || a.elempack != input_pack || !spatial_shape || !flat_ok)
    {
        fprintf(stderr, "test_convolution_int8_flattened failed packing=%d input_int8=%d requant=%d pad=%d ret=%d flat_ret=%d dims=%d w=%d elempack=%d\n", packing, input_int8, requant, pad, ret, flat_ret,
                out.dims, out.w, out.elempack);
        return -1;
    }

    return 0;
}
#endif // NCNN_INT8 && x86

int main()
{
#if NCNN_INT8 && (defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64))
    SRAND(7767517);

    const int pads[] = {0, -233, -234, 1};
    for (int packing = 0; packing < 2; packing++)
    {
        for (int input_int8 = 0; input_int8 < 2; input_int8++)
        {
            for (int requant = 0; requant < 2; requant++)
            {
                for (size_t i = 0; i < sizeof(pads) / sizeof(pads[0]); i++)
                {
                    if (test_convolution_int8_flattened(packing, input_int8, requant, pads[i]) != 0)
                        return -1;
                }
            }
        }
    }
#endif // NCNN_INT8 && x86
    return 0;
}
