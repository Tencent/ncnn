// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_convolution(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, bool use_sgemm, bool use_winograd)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, kernel);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);
    pd.set(6, outch * c * kernel * kernel);

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * c * kernel * kernel);
    if (bias)
        weights[1] = RandomMat(outch);

    float epsilon = 0.001;

    // fp32 path (no bf16)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_packed = false;
        opt.use_bf16_storage = false;
        opt.use_sgemm_convolution = use_sgemm;
        opt.use_winograd_convolution = use_winograd;

        int ret = test_layer_opt("Convolution", pd, weights, opt, a, epsilon);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolution failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d sgemm=%d winograd=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, use_sgemm, use_winograd, activation_type, activation_params[0], activation_params[1]);
            return ret;
        }
    }

    // bf16 path
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_packed = true;
        opt.use_bf16_storage = true;
        opt.use_sgemm_convolution = use_sgemm;
        opt.use_winograd_convolution = use_winograd;

        int ret = test_layer_opt("Convolution", pd, weights, opt, a, epsilon);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolution failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d sgemm=%d winograd=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, use_sgemm, use_winograd, activation_type, activation_params[0], activation_params[1]);
            return ret;
        }
    }

    return 0;
}

// BF16 winograd path: kernel=3, dilation=1, stride=1, winograd=true, sgemm=false
// Need num_input>8 || num_output>8 for prefer_winograd
// winograd43 is the default variant (when neither 63 nor 23 is preferred)
// winograd63: num_input<64 and specific size ranges (e.g. c=16,outch=16,minwh~25-44)
// winograd23: large channels with small spatial (e.g. c=512,outch=512,minwh=3-14)
static int test_convolution_winograd()
{
    // winograd43: default path for moderate sizes
    // c=16,outch=16 => neither prefer_winograd63 nor prefer_winograd23 for w=11,h=10
    int ret = 0
              // Various elempack/out_elempack combos for winograd43
              || test_convolution(11, 10, 16, 16, 3, 1, 1, 1, 1, false, true)
              || test_convolution(11, 10, 16, 24, 3, 1, 1, 1, 0, false, true)
              || test_convolution(11, 10, 24, 16, 3, 1, 1, 1, 1, false, true)
              || test_convolution(11, 10, 16, 3, 3, 1, 1, 1, 0, false, true)
              || test_convolution(11, 10, 3, 16, 3, 1, 1, 1, 1, false, true)
              || test_convolution(11, 10, 16, 1, 3, 1, 1, 1, 0, false, true)
              || test_convolution(11, 10, 1, 16, 3, 1, 1, 1, 1, false, true)
              || test_convolution(11, 10, 16, 2, 3, 1, 1, 1, 0, false, true)
              || test_convolution(11, 10, 2, 16, 3, 1, 1, 1, 1, false, true)
              || test_convolution(11, 10, 9, 9, 3, 1, 1, 1, 0, false, true)
              // out_elempack=8 (outch%16!=0 && outch%8==0)
              || test_convolution(11, 10, 16, 8, 3, 1, 1, 1, 1, false, true)
              // out_elempack=4 (outch%8!=0 && outch%4==0)
              || test_convolution(11, 10, 16, 4, 3, 1, 1, 1, 0, false, true)
              || test_convolution(11, 10, 16, 12, 3, 1, 1, 1, 1, false, true)

              // winograd63: c=16,outch=16, larger spatial => minwh in [23..44]
              || test_convolution(30, 30, 16, 16, 3, 1, 1, 1, 1, false, true)
              || test_convolution(30, 30, 16, 24, 3, 1, 1, 1, 0, false, true)
              || test_convolution(30, 30, 24, 16, 3, 1, 1, 1, 1, false, true)
              || test_convolution(30, 30, 16, 3, 3, 1, 1, 1, 0, false, true)
              || test_convolution(30, 30, 3, 16, 3, 1, 1, 1, 1, false, true)
              // winograd63 with out_elempack=8
              || test_convolution(30, 30, 16, 8, 3, 1, 1, 1, 0, false, true)
              // winograd63 with out_elempack=4 (covers the uncovered winograd63 output transform)
              // Need larger spatial for outch<16 to prefer winograd63 (minwh in [47..128] for num_output>=8)
              || test_convolution(50, 50, 16, 12, 3, 1, 1, 1, 1, false, true)
              // outch=20 (out_elempack=4, outch>=16 for 16-wide output transform)
              || test_convolution(50, 50, 16, 20, 3, 1, 1, 1, 0, false, true)
              || test_convolution(30, 30, 16, 4, 3, 1, 1, 1, 1, false, true)
              || test_convolution(30, 30, 16, 12, 3, 1, 1, 1, 0, false, true)

              // winograd23: large channels, small spatial
              || test_convolution(5, 5, 64, 64, 3, 1, 1, 1, 1, false, true)
              || test_convolution(5, 5, 64, 32, 3, 1, 1, 1, 0, false, true)
              || test_convolution(5, 5, 32, 64, 3, 1, 1, 1, 1, false, true)
              // winograd23 with out_elempack=4
              || test_convolution(5, 5, 64, 4, 3, 1, 1, 1, 0, false, true);

    if (ret != 0)
        return -1;

    return 0;
}

// BF16 im2col_gemm (sgemm) path: sgemm=true, winograd=false
// Triggered when prefer_sgemm=true (large matrix) or kernel=1x1
// prefer_sgemm: num_input * num_output * k*k * dilation^2 * stride^2 * sizeof(bf16) * 2 > L2
//   or num_input > 16 || num_output > 16
static int test_convolution_sgemm()
{
    // 1x1 convolution always goes to sgemm path
    int ret = 0
              || test_convolution(11, 10, 16, 16, 1, 1, 1, 0, 1, true, false)
              || test_convolution(11, 10, 16, 24, 1, 1, 1, 0, 0, true, false)
              || test_convolution(11, 10, 24, 16, 1, 1, 1, 0, 1, true, false)
              || test_convolution(11, 10, 16, 3, 1, 1, 1, 0, 0, true, false)
              || test_convolution(11, 10, 3, 16, 1, 1, 1, 0, 1, true, false)
              || test_convolution(11, 10, 16, 1, 1, 1, 1, 0, 0, true, false)
              || test_convolution(11, 10, 1, 16, 1, 1, 1, 0, 1, true, false)
              || test_convolution(11, 10, 16, 2, 1, 1, 1, 0, 0, true, false)
              || test_convolution(11, 10, 2, 16, 1, 1, 1, 0, 1, true, false)
              || test_convolution(11, 10, 9, 9, 1, 1, 1, 0, 0, true, false)
              || test_convolution(11, 10, 1, 1, 1, 1, 1, 0, 1, true, false)
              // out_elempack=8 (1x1 with outch%8==0, outch%16!=0)
              || test_convolution(11, 10, 16, 8, 1, 1, 1, 0, 1, true, false)
              // out_elempack=4 (1x1 with outch%4==0, outch%8!=0)
              || test_convolution(11, 10, 16, 4, 1, 1, 1, 0, 0, true, false)
              || test_convolution(11, 10, 16, 12, 1, 1, 1, 0, 1, true, false)

              // non-1x1 with prefer_sgemm (num_output > 16)
              || test_convolution(11, 10, 16, 20, 3, 1, 1, 1, 1, true, false)
              || test_convolution(11, 10, 20, 16, 3, 1, 2, 1, 0, true, false)
              || test_convolution(11, 10, 16, 20, 5, 1, 1, 2, 1, true, false)
              || test_convolution(11, 10, 20, 16, 3, 2, 1, 2, 0, true, false)
              || test_convolution(11, 10, 3, 20, 3, 1, 1, 1, 1, true, false)
              || test_convolution(11, 10, 20, 3, 3, 1, 1, 1, 0, true, false)
              // non-1x1 with out_elempack=8
              || test_convolution(11, 10, 16, 8, 3, 1, 1, 1, 1, true, false)
              // non-1x1 with out_elempack=4
              || test_convolution(11, 10, 16, 4, 3, 1, 1, 1, 0, true, false);

    if (ret != 0)
        return -1;

    return 0;
}

// BF16 packed path: sgemm=false, winograd=false (or conditions not met for winograd)
// Falls through to convolution_packed_bf16s
// Kernel configs: A={3,1,2,1} B={5,1,1,-234} C={3,2,1,-234} are spread across test items
static int test_convolution_packed()
{
    return 0
           // out_elempack=16                  k  d  s  p
           || test_convolution(11, 10, 16, 16, 3, 1, 2, 1, 1, false, false) // ep=16
           || test_convolution(11, 10, 24, 16, 5, 1, 1, -234, 0, false, false) // ep=8
           || test_convolution(11, 10,  1, 16, 3, 2, 1, -234, 1, false, false) // ep=1
           || test_convolution(11, 10,  3, 16, 3, 1, 2, 1, 0, false, false) // ep=1
           || test_convolution(11, 10,  4, 16, 5, 1, 1, -234, 1, false, false) // ep=4, kernel_tm inch=4

           // out_elempack=8
           || test_convolution(11, 10, 16,  8, 3, 2, 1, -234, 0, false, false) // ep=16
           || test_convolution(11, 10, 24,  8, 3, 1, 2, 1, 1, false, false) // ep=8, q+15 ep==8
           || test_convolution(11, 10, 20,  8, 5, 1, 1, -234, 0, false, false) // ep=4, q+15 ep==4
           || test_convolution(11, 10, 17,  8, 3, 2, 1, -234, 1, false, false) // ep=1, q+15 ep==1
           || test_convolution(11, 10,  1,  8, 3, 1, 2, 1, 0, false, false) // ep=1, kernel_tm inch=1

           // out_elempack=4
           || test_convolution(11, 10, 16,  4, 5, 1, 1, -234, 0, false, false) // ep=16
           || test_convolution(11, 10, 24,  4, 3, 2, 1, -234, 1, false, false) // ep=8, q+15 ep==8
           || test_convolution(11, 10, 20,  4, 3, 1, 2, 1, 0, false, false) // ep=4, q+15 ep==4
           || test_convolution(11, 10, 17,  4, 5, 1, 1, -234, 1, false, false) // ep=1, q+15 ep==1
           || test_convolution(11, 10,  8,  4, 3, 2, 1, -234, 0, false, false) // ep=8, q+7 ep==8
           || test_convolution(11, 10, 12,  4, 3, 1, 2, 1, 1, false, false) // ep=4, q+7/q+3 ep==4
           || test_convolution(11, 10,  9,  4, 5, 1, 1, -234, 0, false, false) // ep=1, q+7 ep==1
           || test_convolution(11, 10,  5,  4, 3, 2, 1, -234, 1, false, false) // ep=1, q+3 ep==1
           || test_convolution(11, 10,  2,  4, 3, 1, 2, 1, 0, false, false) // kernel_tm inch=2
           || test_convolution(11, 10, 16, 12, 5, 1, 1, -234, 1, false, false) // outch=12

           // out_elempack=1
           || test_convolution(11, 10, 16,  3, 3, 2, 1, -234, 1, false, false) // ep=16
           || test_convolution(11, 10, 24,  1, 3, 1, 2, 1, 0, false, false) // ep=8, q+15 ep==8
           || test_convolution(11, 10, 24,  3, 5, 1, 1, -234, 1, false, false) // ep=8, outch=2+1
           || test_convolution(11, 10, 20,  1, 5, 1, 1, -234, 0, false, false) // ep=4, q+15 ep==4
           || test_convolution(11, 10, 17,  1, 3, 2, 1, -234, 1, false, false) // ep=1, q+15 ep==1
           || test_convolution(11, 10, 12,  1, 3, 1, 2, 1, 0, false, false) // ep=4, q+7 ep==4
           || test_convolution(11, 10,  9,  1, 5, 1, 1, -234, 1, false, false); // ep=1, q+7 ep==1
}

int main()
{
    SRAND(7767517);

    return test_convolution_winograd() || test_convolution_sgemm() || test_convolution_packed();
}
