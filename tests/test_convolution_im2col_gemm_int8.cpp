// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#if NCNN_INT8 && __riscv
#include "layer_type.h"
#include "riscv/convolution_riscv.h"
#include "riscv/riscv_usability.h"

namespace ncnn {
#include "riscv/convolution_im2col_gemm_int8.h"
}
#if NCNN_RUNTIME_CPU && NCNN_RVV
#include "layer/riscv/convolution_riscv_rvv.h"
#endif
#if NCNN_RUNTIME_CPU && NCNN_XTHEADVECTOR
#include "layer/riscv/convolution_riscv_xtheadvector.h"
#endif

struct ConvolutionGeometry
{
    int w, h;
    int kw, kh;
    int dw, dh;
    int sw, sh;
    int left, right, top, bottom;
};

static int test_count = 0;
static int packed_input_count = 0;

// Inspect the actual CPU factory result, so a direct-path fallback cannot make
// a GEMM test pass. Runtime dispatch creates a separate generated RVV class.
template<typename T>
static bool check_pipeline(const ncnn::Layer* layer, bool sgemm, bool lightmode, bool require_split, bool require_m_split)
{
    const T* convolution = static_cast<const T*>(layer);
    if (convolution->weight_sgemm_data.empty() == sgemm
            || convolution->weight_data_tm.empty() != sgemm
            || (lightmode && !convolution->weight_data.empty()))
        return false;

    if (sgemm && require_split && convolution->weight_sgemm_data.h <= 1)
    {
        fprintf(stderr, "INT8 convolution expected split K but got %d transformed K tiles\n", convolution->weight_sgemm_data.h);
        return false;
    }

    if (sgemm && require_m_split && convolution->weight_sgemm_data.c <= 1)
    {
        fprintf(stderr, "INT8 convolution expected split M but got %d transformed M tiles\n", convolution->weight_sgemm_data.c);
        return false;
    }

    return true;
}

static bool check_pipeline(const ncnn::Layer* layer, bool sgemm, bool lightmode, bool require_split, bool require_m_split)
{
#if NCNN_RUNTIME_CPU && NCNN_XTHEADVECTOR
    if (ncnn::cpu_support_riscv_xtheadvector())
        return check_pipeline<ncnn::Convolution_riscv_xtheadvector>(layer, sgemm, lightmode, require_split, require_m_split);
#endif
#if NCNN_RUNTIME_CPU && NCNN_RVV
    if (ncnn::cpu_support_riscv_v())
        return check_pipeline<ncnn::Convolution_riscv_rvv>(layer, sgemm, lightmode, require_split, require_m_split);
#endif
    return check_pipeline<ncnn::Convolution_riscv>(layer, sgemm, lightmode, require_split, require_m_split);
}

static int run_convolution(const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Mat& input, ncnn::Mat& output, const ncnn::Option& opt, int input_pack, bool naive, int forward_threads = 0, bool require_split = false, bool require_m_split = false)
{
    ncnn::Layer* layer = naive ? ncnn::create_layer_naive(ncnn::LayerType::Convolution) : ncnn::create_layer_cpu(ncnn::LayerType::Convolution);
    if (!layer)
        return -1;

    ncnn::ModelBinFromMatArray mb(weights.data());
    int ret = layer->load_param(pd);
    if (ret == 0)
        ret = layer->load_model(mb);
    if (ret == 0)
        ret = layer->create_pipeline(opt);
    if (ret == 0 && !naive && !check_pipeline(layer, opt.use_sgemm_convolution, opt.lightmode, require_split, require_m_split))
    {
        fprintf(stderr, "INT8 convolution selected the wrong pipeline sgemm=%d\n", opt.use_sgemm_convolution);
        ret = -1;
    }

    ncnn::Mat packed_input = input;
    if (ret == 0 && input_pack != 1)
    {
        ncnn::convert_packing(input, packed_input, input_pack, opt);
        if (packed_input.empty() || packed_input.elempack != input_pack)
        {
            fprintf(stderr, "INT8 convolution failed to pack input to %d\n", input_pack);
            ret = -1;
        }
        else
        {
            packed_input_count++;
        }
    }
    if (ret == 0)
    {
        ncnn::Option forward_opt = opt;
        if (forward_threads > 0)
            forward_opt.num_threads = forward_threads;
        ret = layer->forward(packed_input, output, forward_opt);
    }
    if (ret == 0 && output.empty())
        ret = -1;
    if (ret == 0 && output.elempack != 1)
    {
        ncnn::Mat unpacked;
        ncnn::convert_packing(output, unpacked, 1, opt);
        output = unpacked;
        if (output.empty())
            ret = -1;
    }

    layer->destroy_pipeline(opt);
    delete layer;
    return ret;
}

static int compare_int8(const ncnn::Mat& a, const ncnn::Mat& b, int tolerance)
{
    if (a.dims != b.dims || a.w != b.w || a.h != b.h || a.c != b.c || a.elemsize != 1u || b.elemsize != 1u)
        return -1;
    for (int q = 0; q < a.c; q++)
    {
        const signed char* aptr = a.channel(q);
        const signed char* bptr = b.channel(q);
        for (int i = 0; i < a.w * a.h; i++)
        {
            if (abs((int)aptr[i] - (int)bptr[i]) > tolerance)
            {
                fprintf(stderr, "INT8 output mismatch channel=%d index=%d expected=%d actual=%d tolerance=%d\n", q, i, aptr[i], bptr[i], tolerance);
                return -1;
            }
        }
    }
    return 0;
}

static int test_convolution(const ConvolutionGeometry& g, int inch, int outch, int packing, int mode, int activation, int num_threads = 1, int forward_threads = 0, bool require_split = false, bool require_m_split = false)
{
    // mode 0: unit scales/no bias or activation expose exact integer sums;
    // mode 1: dequantize; mode 2: requantize; mode 3: FP32 input quantization.
    const bool exact = mode == 0 || mode == 4;
    const bool requant = mode == 2;
    const bool bias = !exact && (test_count % 2 == 0);
    const int weight_size = inch * outch * g.kw * g.kh;
    const int packn_s8 = ncnn::cpu_riscv_vlenb();
    const int packn = packn_s8 / 4;

    ncnn::Mat input(g.w, g.h, inch, (size_t)(mode == 3 ? 4u : 1u));
    static const int values[] = {-128, -127, -63, -1, 0, 1, 63, 126, 127};
    for (int q = 0; q < inch; q++)
    {
        for (int i = 0; i < g.w * g.h; i++)
        {
            int v = mode == 4 ? (q + i) % 3 - 1 : values[(q * 13 + i * 7 + test_count) % 9];
            if (mode == 3)
                input.channel(q)[i] = (float)(v == -128 ? -127 : v) / 16.f;
            else
                input.channel(q).row<signed char>(0)[i] = (signed char)v;
        }
    }

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, g.kw);
    pd.set(11, g.kh);
    pd.set(2, g.dw);
    pd.set(12, g.dh);
    pd.set(3, g.sw);
    pd.set(13, g.sh);
    pd.set(4, g.left);
    pd.set(15, g.right);
    pd.set(14, g.top);
    pd.set(16, g.bottom);
    pd.set(5, (int)bias);
    pd.set(6, weight_size);
    pd.set(8, requant ? 101 : 1);
    pd.set(9, activation);
    ncnn::Mat activation_params(2);
    activation_params[0] = activation == 6 ? 0.2f : (activation == 3 ? -0.5f : 0.1f);
    activation_params[1] = activation == 3 ? 0.5f : 0.3f;
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights;
    ncnn::Mat kernel(weight_size, (size_t)1u);
    for (int i = 0; i < weight_size; i++)
        ((signed char*)kernel)[i] = (signed char)(mode == 4 ? i % 3 - 1 : values[(i * 5 + i / (inch * g.kw * g.kh) * 3) % 9]);
    weights.push_back(kernel);
    if (bias)
    {
        ncnn::Mat bias_data(outch);
        for (int p = 0; p < outch; p++)
            bias_data[p] = (float)(p % 7 - 3) * 0.25f;
        weights.push_back(bias_data);
    }
    ncnn::Mat weight_scales(outch);
    for (int p = 0; p < outch; p++)
        weight_scales[p] = exact ? 1.f : (p % 7 == 1 ? 0.f : (float)(16 << (p % 3)));
    weights.push_back(weight_scales);
    ncnn::Mat input_scale(1);
    input_scale[0] = exact ? 1.f : 16.f;
    weights.push_back(input_scale);
    if (requant)
    {
        ncnn::Mat output_scale(1);
        output_scale[0] = 2.f;
        weights.push_back(output_scale);
    }

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.lightmode = true;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_winograd_convolution = false;
    opt.use_sgemm_convolution = false;

    ncnn::Mat reference;
    int ret = run_convolution(pd, weights, input, reference, opt, 1, true);
    ncnn::Mat direct;
    ncnn::Mat gemm;
    opt.num_threads = num_threads;
    opt.use_packing_layout = packing != 0;
    int input_pack = 1;
    if (packing == 1 && packn_s8 > 0)
    {
        const int desired_pack = mode == 3 ? packn : packn_s8;
        if (inch % desired_pack == 0)
            input_pack = desired_pack;
    }
    if (ret == 0)
        ret = run_convolution(pd, weights, input, direct, opt, input_pack, false, forward_threads, require_split, require_m_split);
    opt.use_sgemm_convolution = true;
    if (ret == 0)
        ret = run_convolution(pd, weights, input, gemm, opt, input_pack, false, forward_threads, require_split, require_m_split);
    if (ret == 0)
    {
        // Direct and GEMM use identical postprocessing and must match exactly.
        // Naive requantization may differ by one at an FP32 rounding boundary.
        if (requant)
            ret = compare_int8(reference, gemm, 1) || compare_int8(direct, gemm, 0);
        else
        {
            const int naive_ret = CompareMat(reference, gemm, exact ? 0.f : 0.001f);
            const int direct_ret = CompareMat(direct, gemm, 0.f);
            if (naive_ret || direct_ret)
                fprintf(stderr, "GEMM comparison failures naive=%d direct=%d\n", naive_ret, direct_ret);
            ret = naive_ret || direct_ret;
        }
    }
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution_im2col_gemm_int8 failed case=%d w=%d h=%d inch=%d outch=%d kernel=%dx%d dilation=%dx%d stride=%dx%d pad=%d,%d,%d,%d packing=%d input_pack=%d mode=%d bias=%d activation=%d threads=%d\n", test_count, g.w, g.h, inch, outch, g.kw, g.kh, g.dw, g.dh, g.sw, g.sh, g.left, g.right, g.top, g.bottom, packing, input_pack, mode, bias, activation, num_threads);
        return -1;
    }
    test_count++;
    return 0;
}

static int raw_test_count = 0;

static int test_raw_tile(int N, int output_pack, int max_ii)
{
    // Values exceed FP32's exact integer range, so compare raw INT32 storage.
    // Three deliberately unequal K panels exercise initialization and reload.
    const int K = 1057;
#if __riscv_vector
    const int packn = ncnn::cpu_riscv_vlenb();
#else
    const int packn = 1;
#endif
    const int first_channel = packn;
    const int channels = first_channel + max_ii + packn;
    const int width = N + 2;
    const int sentinel = 0x5a5a5a5a;
    ncnn::Mat output(width, 1, channels / output_pack, (size_t)(4u * output_pack), output_pack);
    for (int q = 0; q < output.c; q++)
    {
        int* ptr = output.channel(q);
        for (size_t i = 0; i < output.cstep * output_pack; i++)
            ptr[i] = sentinel;
    }
    std::vector<int> expected(max_ii * N, 0);
    static const int panel_sizes[] = {1, 1024, 32};
    int offset = 0;
    for (int panel = 0; panel < 3; panel++)
    {
        const int count = panel_sizes[panel];
        std::vector<signed char> a(max_ii * count);
        std::vector<signed char> b(count * N);
        int ai = 0;
        for (int ii = 0; ii < max_ii; ii += packn)
        {
            const int lanes = std::min(packn, max_ii - ii);
            for (int k = 0; k < count; k++)
            {
                for (int lane = 0; lane < lanes; lane++)
                {
                    const int m = ii + lane;
                    a[ai++] = (signed char)(m % 2 ? -128 + m % 7 : (offset + k == K - 1 ? 126 : 127) - m % 5);
                }
            }
        }
        int bi = 0;
        int jj = 0;
        while (jj < N)
        {
            const int nn = N - jj >= 4 ? 4 : N - jj >= 2 ? 2 : 1;
            for (int k = 0; k < count; k++)
            {
                for (int n = jj; n < jj + nn; n++)
                    b[bi++] = (signed char)(n % 2 ? -128 + n % 7 : 127 - n % 5);
            }
            jj += nn;
        }
        ncnn::convolution_gemm_transB_packed_tile_int8_rvv(a.data(), b.data(), output, first_channel, max_ii, 1, N, offset, count);
        for (int m = 0; m < max_ii; m++)
        {
            for (int n = 0; n < N; n++)
            {
                for (int k = 0; k < count; k++)
                {
                    const int av = m % 2 ? -128 + m % 7 : (offset + k == K - 1 ? 126 : 127) - m % 5;
                    const int bv = n % 2 ? -128 + n % 7 : 127 - n % 5;
                    expected[m * N + n] += av * bv;
                }
            }
        }
        for (int q = 0; q < channels; q++)
        {
            const int* ptr = output.channel(q / output_pack);
            for (int n = 0; n < width; n++)
            {
                const int actual = ptr[n * output_pack + q % output_pack];
                const int reference = q >= first_channel && q < first_channel + max_ii && n >= 1 && n <= N ? expected[(q - first_channel) * N + n - 1] : sentinel;
                if (actual != reference)
                {
                    fprintf(stderr, "raw INT32 GEMM mismatch N=%d M=%d pack=%d panel=%d q=%d n=%d expected=%d actual=%d\n", N, max_ii, output_pack, panel, q, n, reference, actual);
                    return -1;
                }
            }
        }
        offset += count;
    }
    raw_test_count++;
    return 0;
}

static int test_raw_accumulators()
{
#if __riscv_vector
    const int packn = ncnn::cpu_riscv_vlenb();
#else
    const int packn = 1;
#endif
    const int tails[] = {1, std::max(1, packn - 1), packn, packn + 1, packn * 2 - 1, packn * 2 + 1, packn + 3};
    // Exercise individual blocks, mixed tails and repeated 4-position blocks.
    const int spatial_sizes[] = {1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31};
    for (int n = 0; n < 11; n++)
    {
        for (int i = 0; i < 7; i++)
        {
            if (test_raw_tile(spatial_sizes[n], 1, tails[i]))
                return -1;
        }
        const int output_pack = std::max(1, packn / 4);
        const int packed_tails[] = {output_pack, output_pack * 2, output_pack * 3, packn, packn + output_pack, packn * 2};
        for (int i = 0; i < 6; i++)
        {
            if (test_raw_tile(spatial_sizes[n], output_pack, packed_tails[i]))
                return -1;
        }
    }
    return 0;
}

static int test_geometry()
{
    const int packn_s8 = ncnn::cpu_riscv_vlenb() > 0 ? ncnn::cpu_riscv_vlenb() : 16;
    const int packn = packn_s8 / 4;
    static const ConvolutionGeometry geometries[] = {
        {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
        {13, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
        {9, 7, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1},
        {13, 11, 2, 3, 1, 2, 3, 2, 1, 2, 0, 3},
        {9, 7, 5, 2, 2, 3, 2, 1, -233, -233, -233, -233},
        {8, 9, 3, 2, 1, 2, 2, 3, -234, -234, -234, -234}
    };
    const int channels[][2] = {{1, 1}, {3, 5}, {packn + 1, packn - 1}, {packn_s8, packn}, {packn_s8, packn_s8}, {packn_s8 + 1, packn_s8 + 3}};
    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            for (int packing = 0; packing < 2; packing++)
            {
                if (test_convolution(geometries[i], channels[j][0], channels[j][1], packing, 0, 0) != 0)
                    return -1;
            }
        }
    }
    return 0;
}

static int test_postprocessing()
{
    const int packn_s8 = ncnn::cpu_riscv_vlenb() > 0 ? ncnn::cpu_riscv_vlenb() : 16;
    const ConvolutionGeometry geometry = {9, 5, 2, 2, 1, 1, 2, 1, 1, 0, 0, 2};
    for (int mode = 1; mode <= 3; mode++)
    {
        for (int activation = 0; activation <= 6; activation++)
        {
            if (test_convolution(geometry, packn_s8, packn_s8, 1, mode, activation, activation % 2 + 1) != 0)
                return -1;
            // Packing enabled with actual pack1 input exercises the alternate
            // input layout independently of pipeline packing decisions.
            if (test_convolution(geometry, mode == 3 ? packn_s8 + 1 : packn_s8, packn_s8 + 1, 2, mode, activation) != 0)
                return -1;
        }
    }
    return 0;
}

static int test_split_k()
{
    const int packn = ncnn::cpu_riscv_vlenb() > 0 ? ncnn::cpu_riscv_vlenb() / 4 : 1;
    const int outch = packn * 2 + 1;
    // Query the actual candidate K: a balanced tile from a larger K can be
    // smaller than the cache limit, so tile_k + 1 need not cause a split.
    // The scalar helper gives a conservative (at least as large) bound when
    // this test TU is scalar but the CPU factory dispatches to an RVV layer.
    int inch = 1;
    for (;;)
    {
        int tile_m, tile_n, tile_k;
        ncnn::convolution_im2col_gemm_get_optimal_tile_mnk_int8_rvv(outch, 3, inch, tile_m, tile_n, tile_k, 2);
        if (tile_k < inch)
            break;
        inch = tile_k * 2 + 1;
    }
    const ConvolutionGeometry geometry = {3, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0};
    // Small values avoid accumulator overflow for machines with large caches.
    // Also require multiple transformed K rows and change forward threads.
    return test_convolution(geometry, inch, outch, 0, 4, 0, 2, 3, true);
}

static int test_channel_tiles()
{
    const int packn = ncnn::cpu_riscv_vlenb() > 0 ? ncnn::cpu_riscv_vlenb() : 1;
    const ConvolutionGeometry geometry = {7, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0};
    // Cross multiple widened M tiles with both packed and unpacked output.
    // Require actual transformed M tiles, independent of tiler heuristics.
    for (int tail = 0; tail < 2; tail++)
    {
        for (int packing = 0; packing < 2; packing++)
        {
            for (int threads = 1; threads <= 3; threads += 2)
            {
                if (test_convolution(geometry, 7, packn * 5 + tail, packing, 0, 0, threads, 0, false, true))
                    return -1;
            }
        }
    }
    return 0;
}

static int test_tile_tails()
{
    // Adjacent odd values cover K tails at common channel block boundaries.
    static const int input_channels[] = {7, 15, 16, 17, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 4097};
    static const int spatial_sizes[] = {1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33};
    const int packn = ncnn::cpu_riscv_vlenb() > 0 ? ncnn::cpu_riscv_vlenb() / 4 : 4;
    for (int i = 0; i < 17; i++)
    {
        const ConvolutionGeometry geometry = {spatial_sizes[i % 13], 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0};
        if (test_convolution(geometry, input_channels[i], packn * 2 + 1, i % 2, 0, 0, i % 3 + 1) != 0)
            return -1;
    }
    return 0;
}
#endif // NCNN_INT8 && __riscv

int main()
{
#if NCNN_INT8 && __riscv
    SRAND(7767517);
    if (test_raw_accumulators() || test_geometry() || test_postprocessing() || test_tile_tails() || test_split_k() || test_channel_tiles())
        return -1;
    fprintf(stderr, "PASS INT8 convolution GEMM: %d raw split-K checks, %d layer cases, %d packed input runs, vlenb=%d\n", raw_test_count, test_count, packed_input_count, ncnn::cpu_riscv_vlenb());
#else
    fprintf(stderr, "SKIP RISC-V INT8 convolution GEMM tests\n");
#endif
    return 0;
}
