// Xavier Hsinyuan is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 Xavier Hsinyuan <me@lstlx.com>. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "gru_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

namespace ncnn {

//core rvv-optimized gru impl.
#if __riscv_vector
static int gru(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& hidden_state, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;

    // 2 x num_output
    Mat gates(2, num_output, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    // unroll
    for (int t = 0; t < T; t++)
    {
        int ti = reverse ? T - 1 - t : t;

        const float* x = bottom_blob.row(ti);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_output; q++)
        {
            float* gates_data = gates.row(q);

            // gate reset update
            const float* bias_c_R = bias_c.row(0);
            const float* bias_c_U = bias_c.row(1);

            const float* weight_xc_R = weight_xc.row(num_output * 0 + q);
            const float* weight_xc_U = weight_xc.row(num_output * 1 + q);
            const float* weight_hc_R = weight_hc.row(num_output * 0 + q);
            const float* weight_hc_U = weight_hc.row(num_output * 1 + q);

            float R = bias_c_R[q];
            float U = bias_c_U[q];

            int n = size;
            const float* ptr_x = x;
            const float* ptr_xcr = weight_xc_R;
            const float* ptr_xcu = weight_xc_U;
            while (n > 0)
            {
                size_t vl = vsetvl_e32m8(n);
                vfloat32m8_t _x = vle32_v_f32m8(ptr_x, vl);
                vfloat32m8_t _xcr = vle32_v_f32m8(ptr_xcr, vl);
                vfloat32m8_t _xcu = vle32_v_f32m8(ptr_xcu, vl);
                vfloat32m1_t _scalar_r = vfmv_s_f_f32m1(vundefined_f32m1(), R, vl);
                vfloat32m1_t _scalar_u = vfmv_s_f_f32m1(vundefined_f32m1(), U, vl);

                _xcr = vfmul_vv_f32m8(_xcr, _x, vl);
                _xcu = vfmul_vv_f32m8(_xcu, _x, vl);
                _scalar_r = vfredusum_vs_f32m8_f32m1(_scalar_r, _xcr, _scalar_r, vl);
                _scalar_u = vfredusum_vs_f32m8_f32m1(_scalar_u, _xcu, _scalar_u, vl);

                R = vfmv_f_s_f32m1_f32(_scalar_r);
                U = vfmv_f_s_f32m1_f32(_scalar_u);

                ptr_x += vl;
                ptr_xcr += vl;
                ptr_xcu += vl;
                n -= vl;
            }
            ptr_x = NULL;
            ptr_xcr = NULL;
            ptr_xcu = NULL;

            int n_out = num_output;
            const float* ptr_hc = hidden_state;
            const float* ptr_hcr = weight_hc_R;
            const float* ptr_hcu = weight_hc_U;
            while (n_out > 0)
            {
                size_t vl = vsetvl_e32m8(n_out);
                vfloat32m8_t _h_cont = vle32_v_f32m8(ptr_hc, vl);
                vfloat32m8_t _hcr = vle32_v_f32m8(ptr_hcr, vl);
                vfloat32m8_t _hcu = vle32_v_f32m8(ptr_hcu, vl);
                vfloat32m1_t _scalar_r = vfmv_s_f_f32m1(vundefined_f32m1(), R, vl);
                vfloat32m1_t _scalar_u = vfmv_s_f_f32m1(vundefined_f32m1(), U, vl);

                _hcr = vfmul_vv_f32m8(_hcr, _h_cont, vl);
                _hcu = vfmul_vv_f32m8(_hcu, _h_cont, vl);
                _scalar_r = vfredusum_vs_f32m8_f32m1(_scalar_r, _hcr, _scalar_r, vl);
                _scalar_u = vfredusum_vs_f32m8_f32m1(_scalar_u, _hcu, _scalar_u, vl);

                R = vfmv_f_s_f32m1_f32(_scalar_r);
                U = vfmv_f_s_f32m1_f32(_scalar_u);

                ptr_hc += vl;
                ptr_hcr += vl;
                ptr_hcu += vl;
                n_out -= vl;
            }
            ptr_hc = NULL;
            ptr_hcr = NULL;
            ptr_hcu = NULL;

            // sigmoid(R)
            // sigmoid(U)
            R = 1.f / (1.f + exp(-R));
            U = 1.f / (1.f + exp(-U));

            // gate new
            const float* bias_c_WN = bias_c.row(2);
            const float* bias_c_BN = bias_c.row(3);

            const float* weight_xc_N = weight_xc.row(num_output * 2 + q);
            const float* weight_hc_N = weight_hc.row(num_output * 2 + q);

            float N = bias_c_BN[q];

            int n_out2 = num_output;
            const float* ptr_hc2 = hidden_state;
            const float* ptr_whc_n = weight_hc_N;
            while (n_out2 > 0)
            {
                size_t vl = vsetvl_e32m8(n_out2);

                vfloat32m8_t _h_cont = vle32_v_f32m8(ptr_hc2, vl);
                vfloat32m8_t _whc_n = vle32_v_f32m8(ptr_whc_n, vl);
                vfloat32m1_t _scalar_n = vfmv_s_f_f32m1(vundefined_f32m1(), N, vl);

                _h_cont = vfmul_vv_f32m8(_whc_n, _h_cont, vl);
                _scalar_n = vfredusum_vs_f32m8_f32m1(_scalar_n, _h_cont, _scalar_n, vl);

                N = vfmv_f_s_f32m1_f32(_scalar_n);
                n_out2 -= vl;
                ptr_hc2 += vl;
                ptr_whc_n += vl;
            }
            ptr_hc2 = NULL;
            ptr_whc_n = NULL;

            N = bias_c_WN[q] + R * N;

            int n2 = size;
            const float* ptr_x2 = x;
            const float* ptr_xcn = weight_xc_N;
            while (n2 > 0)
            {
                size_t vl = vsetvl_e32m8(n2);

                vfloat32m8_t _x = vle32_v_f32m8(ptr_x2, vl);
                vfloat32m8_t _xcn = vle32_v_f32m8(ptr_xcn, vl);
                vfloat32m1_t _scalar_n = vfmv_s_f_f32m1(vundefined_f32m1(), N, vl);

                _xcn = vfmul_vv_f32m8(_x, _xcn, vl);
                _scalar_n = vfredusum_vs_f32m8_f32m1(_scalar_n, _xcn, _scalar_n, vl);
                N = vfmv_f_s_f32m1_f32(_scalar_n);

                n2 -= vl;
                ptr_x2 += vl;
                ptr_xcn += vl;
            }
            ptr_x2 = NULL;
            ptr_xcn = NULL;

            // tanh(N)
            N = tanh(N);

            gates_data[0] = U;
            gates_data[1] = N;
        }

        // h_t := (1 - update) .* new + update .* h_{t-1}
        float* output_data = top_blob.row(ti);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_output; q++)
        {
            const float* gates_data = gates.row(q);

            float U = gates_data[0];
            float N = gates_data[1];

            float H = (1 - U) * N + U * hidden_state[q];

            hidden_state[q] = H;
            output_data[q] = H;
        }
    }

    return 0;
}

#endif

GRU_riscv::GRU_riscv()
{
#if __riscv_vector && __riscv_zfh
    support_fp16_storage = true;
#endif
}

int GRU_riscv::create_pipeline(const Option& opt)
{
#if __riscv_vector && __riscv_zfh
    if (opt.use_fp16_storage && opt.use_fp16_arithmetic)
        return create_pipeline_fp16sa(opt);
#endif

    return GRU::create_pipeline(opt);
}

int GRU_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();
#if __riscv_vector

#if __riscv_zfh
    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blob, top_blob, opt);
        else
            return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

    int T = bottom_blob.h;

    int num_directions = direction == 2 ? 2 : 1;

    // initial hidden state
    Mat hidden(num_output, 4u, opt.workspace_allocator);
    if (hidden.empty())
        return -100;
    hidden.fill(0.f);

    top_blob.create(num_output * num_directions, T, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
        int ret = gru(bottom_blob, top_blob, direction, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden, opt);
        if (ret != 0)
            return ret;
    }

    if (direction == 2)
    {
        Mat top_blob_forward(num_output, T, 4u, opt.workspace_allocator);
        if (top_blob_forward.empty())
            return -100;

        Mat top_blob_reverse(num_output, T, 4u, opt.workspace_allocator);
        if (top_blob_reverse.empty())
            return -100;

        int ret0 = gru(bottom_blob, top_blob_forward, 0, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden, opt);
        if (ret0 != 0)
            return ret0;

        hidden.fill(0.0f);

        int ret1 = gru(bottom_blob, top_blob_reverse, 1, weight_xc_data.channel(1), bias_c_data.channel(1), weight_hc_data.channel(1), hidden, opt);
        if (ret1 != 0)
            return ret1;

        // concat w
        for (int i = 0; i < T; i++)
        {
            const float* pf = top_blob_forward.row(i);
            const float* pr = top_blob_reverse.row(i);
            float* ptr = top_blob.row(i);

            memcpy(ptr, pf, num_output * sizeof(float));
            memcpy(ptr + num_output, pr, num_output * sizeof(float));
        }
    }

    return 0;
#endif
    return GRU::forward(bottom_blob, top_blob, opt);
}

int GRU_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int elembits = bottom_blob.elembits();

#if __riscv_vector
#if __riscv_zfh
    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blobs, top_blobs, opt);
        else
            return forward_fp16s(bottom_blobs, top_blobs, opt);
    }
#endif

    int T = bottom_blob.h;
    int num_directions = direction == 2 ? 2 : 1;

    Mat hidden;
    Allocator* hidden_allocator = top_blobs.size() == 2 ? opt.blob_allocator : opt.workspace_allocator;
    if (bottom_blobs.size() == 2)
    {
        hidden = bottom_blobs[1].clone(hidden_allocator);
    }
    else
    {
        hidden.create(num_output, num_directions, 4u, hidden_allocator);
        if (hidden.empty())
            return -100;
        hidden.fill(0.f);
    }

    Mat& top_blob = top_blobs[0];
    top_blob.create(num_output * num_directions, T, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
        Mat hidden0 = hidden.row_range(0, 1);
        int ret = gru(bottom_blob, top_blob, direction, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden0, opt);
        if (ret != 0)
            return ret;
    }

    if (direction == 2)
    {
        Mat top_blob_forward(num_output, T, 4u, opt.workspace_allocator);
        if (top_blob_forward.empty())
            return -100;

        Mat top_blob_reverse(num_output, T, 4u, opt.workspace_allocator);
        if (top_blob_reverse.empty())
            return -100;

        Mat hidden0 = hidden.row_range(0, 1);
        int ret0 = gru(bottom_blob, top_blob_forward, 0, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden0, opt);
        if (ret0 != 0)
            return ret0;

        Mat hidden1 = hidden.row_range(1, 1);
        int ret1 = gru(bottom_blob, top_blob_reverse, 1, weight_xc_data.channel(1), bias_c_data.channel(1), weight_hc_data.channel(1), hidden1, opt);
        if (ret1 != 0)
            return ret1;

        // concat w
        for (int i = 0; i < T; i++)
        {
            const float* pf = top_blob_forward.row(i);
            const float* pr = top_blob_reverse.row(i);
            float* ptr = top_blob.row(i);

            memcpy(ptr, pf, num_output * sizeof(float));
            memcpy(ptr + num_output, pr, num_output * sizeof(float));
        }
    }

    if (top_blobs.size() == 2)
    {
        top_blobs[1] = hidden;
    }

    return 0;
#endif
    return GRU::forward(bottom_blobs, top_blobs, opt);
}

#if __riscv_vector && __riscv_zfh
static int gru_fp16s(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& hidden_state, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;

    // 2 x num_output
    Mat gates(2, num_output, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    // unroll
    for (int t = 0; t < T; t++)
    {
        int ti = reverse ? T - 1 - t : t;

        const __fp16* x = bottom_blob.row<const __fp16>(ti);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_output; q++)
        {
            float* gates_data = gates.row(q);

            // gate reset update
            const float* bias_c_R = bias_c.row(0);
            const float* bias_c_U = bias_c.row(1);

            const float* weight_xc_R = weight_xc.row(num_output * 0 + q);
            const float* weight_xc_U = weight_xc.row(num_output * 1 + q);
            const float* weight_hc_R = weight_hc.row(num_output * 0 + q);
            const float* weight_hc_U = weight_hc.row(num_output * 1 + q);

            float R = bias_c_R[q];
            float U = bias_c_U[q];

            int n = size;
            const __fp16* ptr_x = x;
            const float* ptr_xcr = weight_xc_R;
            const float* ptr_xcu = weight_xc_U;
            while (n > 0)
            {
                size_t vl = vsetvl_e16m4(n);
                vfloat32m8_t _x = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr_x, vl), vl);
                vfloat32m8_t _xcr = vle32_v_f32m8(ptr_xcr, vl);
                vfloat32m8_t _xcu = vle32_v_f32m8(ptr_xcu, vl);
                vfloat32m1_t _scalar_r = vfmv_s_f_f32m1(vundefined_f32m1(), R, vl);
                vfloat32m1_t _scalar_u = vfmv_s_f_f32m1(vundefined_f32m1(), U, vl);

                _xcr = vfmul_vv_f32m8(_xcr, _x, vl);
                _xcu = vfmul_vv_f32m8(_xcu, _x, vl);
                _scalar_r = vfredusum_vs_f32m8_f32m1(_scalar_r, _xcr, _scalar_r, vl);
                _scalar_u = vfredusum_vs_f32m8_f32m1(_scalar_u, _xcu, _scalar_u, vl);

                R = vfmv_f_s_f32m1_f32(_scalar_r);
                U = vfmv_f_s_f32m1_f32(_scalar_u);

                ptr_x += vl;
                ptr_xcr += vl;
                ptr_xcu += vl;
                n -= vl;
            }
            ptr_x = NULL;
            ptr_xcr = NULL;
            ptr_xcu = NULL;

            int n_out = num_output;
            const float* ptr_hc = hidden_state;
            const float* ptr_hcr = weight_hc_R;
            const float* ptr_hcu = weight_hc_U;
            while (n_out > 0)
            {
                size_t vl = vsetvl_e16m4(n_out);
                vfloat32m8_t _h_cont = vle32_v_f32m8(ptr_hc, vl);
                vfloat32m8_t _hcr = vle32_v_f32m8(ptr_hcr, vl);
                vfloat32m8_t _hcu = vle32_v_f32m8(ptr_hcu, vl);
                vfloat32m1_t _scalar_r = vfmv_s_f_f32m1(vundefined_f32m1(), R, vl);
                vfloat32m1_t _scalar_u = vfmv_s_f_f32m1(vundefined_f32m1(), U, vl);

                _hcr = vfmul_vv_f32m8(_hcr, _h_cont, vl);
                _hcu = vfmul_vv_f32m8(_hcu, _h_cont, vl);
                _scalar_r = vfredusum_vs_f32m8_f32m1(_scalar_r, _hcr, _scalar_r, vl);
                _scalar_u = vfredusum_vs_f32m8_f32m1(_scalar_u, _hcu, _scalar_u, vl);

                R = vfmv_f_s_f32m1_f32(_scalar_r);
                U = vfmv_f_s_f32m1_f32(_scalar_u);

                ptr_hc += vl;
                ptr_hcr += vl;
                ptr_hcu += vl;
                n_out -= vl;
            }
            ptr_hc = NULL;
            ptr_hcr = NULL;
            ptr_hcu = NULL;

            // sigmoid(R)
            // sigmoid(U)
            R = 1.f / (1.f + exp(-R));
            U = 1.f / (1.f + exp(-U));

            // gate new
            const float* bias_c_WN = bias_c.row(2);
            const float* bias_c_BN = bias_c.row(3);

            const float* weight_xc_N = weight_xc.row(num_output * 2 + q);
            const float* weight_hc_N = weight_hc.row(num_output * 2 + q);

            float N = bias_c_BN[q];

            int n_out2 = num_output;
            const float* ptr_hc2 = hidden_state;
            const float* ptr_whc_n = weight_hc_N;
            while (n_out2 > 0)
            {
                size_t vl = vsetvl_e16m4(n_out2);

                vfloat32m8_t _h_cont = vle32_v_f32m8(ptr_hc2, vl);
                vfloat32m8_t _whc_n = vle32_v_f32m8(ptr_whc_n, vl);
                vfloat32m1_t _scalar_n = vfmv_s_f_f32m1(vundefined_f32m1(), N, vl);

                _h_cont = vfmul_vv_f32m8(_whc_n, _h_cont, vl);
                _scalar_n = vfredusum_vs_f32m8_f32m1(_scalar_n, _h_cont, _scalar_n, vl);

                N = vfmv_f_s_f32m1_f32(_scalar_n);
                n_out2 -= vl;
                ptr_hc2 += vl;
                ptr_whc_n += vl;
            }
            ptr_hc2 = NULL;
            ptr_whc_n = NULL;

            N = bias_c_WN[q] + R * N;

            int n2 = size;
            const __fp16* ptr_x2 = x;
            const float* ptr_xcn = weight_xc_N;
            while (n2 > 0)
            {
                size_t vl = vsetvl_e16m4(n2);

                vfloat32m8_t _x = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr_x2, vl), vl);
                vfloat32m8_t _xcn = vle32_v_f32m8(ptr_xcn, vl);
                vfloat32m1_t _scalar_n = vfmv_s_f_f32m1(vundefined_f32m1(), N, vl);

                _xcn = vfmul_vv_f32m8(_x, _xcn, vl);
                _scalar_n = vfredusum_vs_f32m8_f32m1(_scalar_n, _xcn, _scalar_n, vl);
                N = vfmv_f_s_f32m1_f32(_scalar_n);

                n2 -= vl;
                ptr_x2 += vl;
                ptr_xcn += vl;
            }
            ptr_x2 = NULL;
            ptr_xcn = NULL;

            // tanh(N)
            N = tanh(N);

            gates_data[0] = U;
            gates_data[1] = N;
        }

        // h_t := (1 - update) .* new + update .* h_{t-1}
        __fp16* output_data = top_blob.row<__fp16>(ti);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_output; q++)
        {
            const float* gates_data = gates.row(q);

            float U = gates_data[0];
            float N = gates_data[1];

            float H = (1 - U) * N + U * hidden_state[q];

            hidden_state[q] = H;
            output_data[q] = (__fp16)H;
        }
    }

    return 0;
}

int GRU_riscv::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int T = bottom_blob.h;

    int num_directions = direction == 2 ? 2 : 1;
    // initial hidden state
    Mat hidden(num_output, 4u, opt.workspace_allocator);
    if (hidden.empty())
        return -100;
    hidden.fill(0.f);

    top_blob.create(num_output * num_directions, T, 2u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
        int ret = gru_fp16s(bottom_blob, top_blob, direction, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden, opt);
        if (ret != 0)
            return ret;
    }

    if (direction == 2)
    {
        Mat top_blob_forward(num_output, T, 2u, opt.workspace_allocator);
        if (top_blob_forward.empty())
            return -100;

        Mat top_blob_reverse(num_output, T, 2u, opt.workspace_allocator);
        if (top_blob_reverse.empty())
            return -100;

        int ret0 = gru_fp16s(bottom_blob, top_blob_forward, 0, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden, opt);
        if (ret0 != 0)
            return ret0;

        hidden.fill(0.0f);

        int ret1 = gru_fp16s(bottom_blob, top_blob_reverse, 1, weight_xc_data.channel(1), bias_c_data.channel(1), weight_hc_data.channel(1), hidden, opt);
        if (ret1 != 0)
            return ret1;

        // concat w
        for (int i = 0; i < T; i++)
        {
            const __fp16* pf = top_blob_forward.row<const __fp16>(i);
            const __fp16* pr = top_blob_reverse.row<const __fp16>(i);
            __fp16* ptr = top_blob.row<__fp16>(i);

            memcpy(ptr, pf, num_output * sizeof(__fp16));
            memcpy(ptr + num_output, pr, num_output * sizeof(__fp16));
        }
    }

    return 0;
}

int GRU_riscv::forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int T = bottom_blob.h;
    int num_directions = direction == 2 ? 2 : 1;

    Mat hidden;
    Allocator* hidden_allocator = top_blobs.size() == 2 ? opt.blob_allocator : opt.workspace_allocator;
    if (bottom_blobs.size() == 2)
    {
        Option opt_cast = opt;
        opt_cast.blob_allocator = hidden_allocator;
        cast_float16_to_float32(bottom_blobs[1], hidden, opt_cast);
    }
    else
    {
        hidden.create(num_output, num_directions, 4u, hidden_allocator);
        if (hidden.empty())
            return -100;
        hidden.fill(0.f);
    }

    Mat& top_blob = top_blobs[0];
    top_blob.create(num_output * num_directions, T, 2u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
        Mat hidden0 = hidden.row_range(0, 1);
        int ret = gru_fp16s(bottom_blob, top_blob, direction, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden0, opt);
        if (ret != 0)
            return ret;
    }

    if (direction == 2)
    {
        Mat top_blob_forward(num_output, T, 2u, opt.workspace_allocator);
        if (top_blob_forward.empty())
            return -100;

        Mat top_blob_reverse(num_output, T, 2u, opt.workspace_allocator);
        if (top_blob_reverse.empty())
            return -100;

        Mat hidden0 = hidden.row_range(0, 1);
        int ret0 = gru_fp16s(bottom_blob, top_blob_forward, 0, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden0, opt);
        if (ret0 != 0)
            return ret0;

        Mat hidden1 = hidden.row_range(1, 1);
        int ret1 = gru_fp16s(bottom_blob, top_blob_reverse, 1, weight_xc_data.channel(1), bias_c_data.channel(1), weight_hc_data.channel(1), hidden1, opt);
        if (ret1 != 0)
            return ret1;

        // concat w
        for (int i = 0; i < T; i++)
        {
            const __fp16* pf = top_blob_forward.row<const __fp16>(i);
            const __fp16* pr = top_blob_reverse.row<const __fp16>(i);
            __fp16* ptr = top_blob.row<__fp16>(i);

            memcpy(ptr, pf, num_output * sizeof(__fp16));
            memcpy(ptr + num_output, pr, num_output * sizeof(__fp16));
        }
    }

    if (top_blobs.size() == 2)
    {
        cast_float32_to_float16(hidden, top_blobs[1], opt);
    }

    return 0;
}

#endif

//fp16sa start at here
#if __riscv_vector && __riscv_zfh

int GRU_riscv::create_pipeline_fp16sa(const Option& opt)
{
    cast_float32_to_float16(weight_xc_data, weight_xc_data_fp16sa, opt);
    cast_float32_to_float16(weight_hc_data, weight_hc_data_fp16sa, opt);
    cast_float32_to_float16(bias_c_data, bias_c_data_fp16sa, opt);

    weight_xc_data.release();
    bias_c_data.release();
    weight_hc_data.release();

    return 0;
}

static int gru_fp16sa(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& hidden_state, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;

    // 2 x num_output
    Mat gates(2, num_output, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    // unroll
    for (int t = 0; t < T; t++)
    {
        int ti = reverse ? T - 1 - t : t;

        const __fp16* x = bottom_blob.row<const __fp16>(ti);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_output; q++)
        {
            float* gates_data = gates.row(q);

            // gate reset update
            const __fp16* bias_c_R = bias_c.row<const __fp16>(0);
            const __fp16* bias_c_U = bias_c.row<const __fp16>(1);

            const __fp16* weight_xc_R = weight_xc.row<const __fp16>(num_output * 0 + q);
            const __fp16* weight_xc_U = weight_xc.row<const __fp16>(num_output * 1 + q);
            const __fp16* weight_hc_R = weight_hc.row<const __fp16>(num_output * 0 + q);
            const __fp16* weight_hc_U = weight_hc.row<const __fp16>(num_output * 1 + q);

            __fp16 R = bias_c_R[q];
            __fp16 U = bias_c_U[q];

            int n = size;
            const __fp16* ptr_x = x;
            const __fp16* ptr_xcr = weight_xc_R;
            const __fp16* ptr_xcu = weight_xc_U;
            while (n > 0)
            {
                size_t vl = vsetvl_e16m8(n);
                vfloat16m8_t _x = vle16_v_f16m8(ptr_x, vl);
                vfloat16m8_t _xcr = vle16_v_f16m8(ptr_xcr, vl);
                vfloat16m8_t _xcu = vle16_v_f16m8(ptr_xcu, vl);
                vfloat16m1_t _scalar_r = vfmv_s_f_f16m1(vundefined_f16m1(), R, vl);
                vfloat16m1_t _scalar_u = vfmv_s_f_f16m1(vundefined_f16m1(), U, vl);

                _xcr = vfmul_vv_f16m8(_xcr, _x, vl);
                _xcu = vfmul_vv_f16m8(_xcu, _x, vl);
                _scalar_r = vfredusum_vs_f16m8_f16m1(_scalar_r, _xcr, _scalar_r, vl);
                _scalar_u = vfredusum_vs_f16m8_f16m1(_scalar_u, _xcu, _scalar_u, vl);

                R = vfmv_f_s_f16m1_f16(_scalar_r);
                U = vfmv_f_s_f16m1_f16(_scalar_u);

                ptr_x += vl;
                ptr_xcr += vl;
                ptr_xcu += vl;
                n -= vl;
            }

            int n_out = num_output;
            const float* ptr_hc = hidden_state;
            const __fp16* ptr_hcr = weight_hc_R;
            const __fp16* ptr_hcu = weight_hc_U;
            while (n_out > 0)
            {
                size_t vl = vsetvl_e16m4(n_out);
                vfloat16m4_t _h_cont = vfncvt_f_f_w_f16m4(vle32_v_f32m8(ptr_hc, vl), vl);
                vfloat16m4_t _hcr = vle16_v_f16m4(ptr_hcr, vl);
                vfloat16m4_t _hcu = vle16_v_f16m4(ptr_hcu, vl);
                vfloat16m1_t _scalar_r = vfmv_s_f_f16m1(vundefined_f16m1(), R, vl);
                vfloat16m1_t _scalar_u = vfmv_s_f_f16m1(vundefined_f16m1(), U, vl);

                _hcr = vfmul_vv_f16m4(_hcr, _h_cont, vl);
                _hcu = vfmul_vv_f16m4(_hcu, _h_cont, vl);
                _scalar_r = vfredusum_vs_f16m4_f16m1(_scalar_r, _hcr, _scalar_r, vl);
                _scalar_u = vfredusum_vs_f16m4_f16m1(_scalar_u, _hcu, _scalar_u, vl);

                R = vfmv_f_s_f16m1_f16(_scalar_r);
                U = vfmv_f_s_f16m1_f16(_scalar_u);

                ptr_hc += vl;
                ptr_hcr += vl;
                ptr_hcu += vl;
                n_out -= vl;
            }

            // sigmoid(R)
            // sigmoid(U)
            R = 1.f / (1.f + (__fp16)exp((float)(-R)));
            U = 1.f / (1.f + (__fp16)exp((float)(-U)));

            // gate new
            const __fp16* bias_c_WN = bias_c.row<const __fp16>(2);
            const __fp16* bias_c_BN = bias_c.row<const __fp16>(3);

            const __fp16* weight_xc_N = weight_xc.row<const __fp16>(num_output * 2 + q);
            const __fp16* weight_hc_N = weight_hc.row<const __fp16>(num_output * 2 + q);

            __fp16 N = bias_c_BN[q];

            int n_out2 = num_output;
            const float* ptr_hc2 = hidden_state;
            const __fp16* ptr_whc_n = weight_hc_N;
            while (n_out2 > 0)
            {
                size_t vl = vsetvl_e16m4(n_out2);

                vfloat16m4_t _h_cont = vfncvt_f_f_w_f16m4(vle32_v_f32m8(ptr_hc2, vl), vl);
                vfloat16m4_t _whc_n = vle16_v_f16m4(ptr_whc_n, vl);
                vfloat16m1_t _scalar_n = vfmv_s_f_f16m1(vundefined_f16m1(), N, vl);

                _h_cont = vfmul_vv_f16m4(_whc_n, _h_cont, vl);
                _scalar_n = vfredusum_vs_f16m4_f16m1(_scalar_n, _h_cont, _scalar_n, vl);

                N = vfmv_f_s_f16m1_f16(_scalar_n);
                n_out2 -= vl;
                ptr_hc2 += vl;
                ptr_whc_n += vl;
            }
            N = bias_c_WN[q] + R * N;

            int n2 = size;
            const __fp16* ptr_x2 = x;
            const __fp16* ptr_xcn = weight_xc_N;
            while (n2 > 0)
            {
                size_t vl = vsetvl_e16m8(n2);

                vfloat16m8_t _x = vle16_v_f16m8(ptr_x2, vl);
                vfloat16m8_t _xcn = vle16_v_f16m8(ptr_xcn, vl);
                vfloat16m1_t _scalar_n = vfmv_s_f_f16m1(vundefined_f16m1(), N, vl);

                _xcn = vfmul_vv_f16m8(_x, _xcn, vl);
                _scalar_n = vfredusum_vs_f16m8_f16m1(_scalar_n, _xcn, _scalar_n, vl);
                N = vfmv_f_s_f16m1_f16(_scalar_n);

                n2 -= vl;
                ptr_x2 += vl;
                ptr_xcn += vl;
            }

            // tanh(N)
            N = (__fp16)tanh((float)N);

            gates_data[0] = U;
            gates_data[1] = N;
        }

        // h_t := (1 - update) .* new + update .* h_{t-1}
        __fp16* output_data = top_blob.row<__fp16>(ti);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_output; q++)
        {
            const float* gates_data = gates.row(q);

            float U = gates_data[0];
            float N = gates_data[1];

            float H = (1 - U) * N + U * hidden_state[q];

            hidden_state[q] = H;
            output_data[q] = H;
        }
    }

    return 0;
}

int GRU_riscv::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int T = bottom_blob.h;

    int num_directions = direction == 2 ? 2 : 1;
    // initial hidden state
    Mat hidden(num_output, 4u, opt.workspace_allocator);
    if (hidden.empty())
        return -100;
    hidden.fill(0.f);

    top_blob.create(num_output * num_directions, T, 2u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
        int ret = gru_fp16sa(bottom_blob, top_blob, direction, weight_xc_data_fp16sa.channel(0), bias_c_data_fp16sa.channel(0), weight_hc_data_fp16sa.channel(0), hidden, opt);
        if (ret != 0)
            return ret;
    }

    if (direction == 2)
    {
        Mat top_blob_forward(num_output, T, 2u, opt.workspace_allocator);
        if (top_blob_forward.empty())
            return -100;

        Mat top_blob_reverse(num_output, T, 2u, opt.workspace_allocator);
        if (top_blob_reverse.empty())
            return -100;

        int ret0 = gru_fp16sa(bottom_blob, top_blob_forward, 0, weight_xc_data_fp16sa.channel(0), bias_c_data_fp16sa.channel(0), weight_hc_data_fp16sa.channel(0), hidden, opt);
        if (ret0 != 0)
            return ret0;

        hidden.fill(0.0f);

        int ret1 = gru_fp16sa(bottom_blob, top_blob_reverse, 1, weight_xc_data_fp16sa.channel(1), bias_c_data_fp16sa.channel(1), weight_hc_data_fp16sa.channel(1), hidden, opt);
        if (ret1 != 0)
            return ret1;

        // concat w
        for (int i = 0; i < T; i++)
        {
            const __fp16* pf = top_blob_forward.row<const __fp16>(i);
            const __fp16* pr = top_blob_reverse.row<const __fp16>(i);
            __fp16* ptr = top_blob.row<__fp16>(i);

            memcpy(ptr, pf, num_output * sizeof(__fp16));
            memcpy(ptr + num_output, pr, num_output * sizeof(__fp16));
        }
    }

    return 0;
}

int GRU_riscv::forward_fp16sa(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int T = bottom_blob.h;
    int num_directions = direction == 2 ? 2 : 1;

    Mat hidden;
    Allocator* hidden_allocator = top_blobs.size() == 2 ? opt.blob_allocator : opt.workspace_allocator;
    if (bottom_blobs.size() == 2)
    {
        Option opt_cast = opt;
        opt_cast.blob_allocator = hidden_allocator;
        cast_float16_to_float32(bottom_blobs[1], hidden, opt_cast);
    }
    else
    {
        hidden.create(num_output, num_directions, 4u, hidden_allocator);
        if (hidden.empty())
            return -100;
        hidden.fill(0.f);
    }

    Mat& top_blob = top_blobs[0];
    top_blob.create(num_output * num_directions, T, 2u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
        int ret = gru_fp16sa(bottom_blob, top_blob, direction, weight_xc_data_fp16sa.channel(0), bias_c_data_fp16sa.channel(0), weight_hc_data_fp16sa.channel(0), hidden, opt);
        if (ret != 0)
            return ret;
    }

    if (direction == 2)
    {
        Mat top_blob_forward(num_output, T, 2u, opt.workspace_allocator);
        if (top_blob_forward.empty())
            return -100;

        Mat top_blob_reverse(num_output, T, 2u, opt.workspace_allocator);
        if (top_blob_reverse.empty())
            return -100;

        Mat hidden0 = hidden.row_range(0, 1);
        int ret0 = gru_fp16sa(bottom_blob, top_blob_forward, 0, weight_xc_data_fp16sa.channel(0), bias_c_data_fp16sa.channel(0), weight_hc_data_fp16sa.channel(0), hidden0, opt);
        if (ret0 != 0)
            return ret0;

        Mat hidden1 = hidden.row_range(1, 1);
        int ret1 = gru_fp16sa(bottom_blob, top_blob_reverse, 1, weight_xc_data_fp16sa.channel(1), bias_c_data_fp16sa.channel(1), weight_hc_data_fp16sa.channel(1), hidden1, opt);
        if (ret1 != 0)
            return ret1;

        // concat w
        for (int i = 0; i < T; i++)
        {
            const __fp16* pf = top_blob_forward.row<const __fp16>(i);
            const __fp16* pr = top_blob_reverse.row<const __fp16>(i);
            __fp16* ptr = top_blob.row<__fp16>(i);

            memcpy(ptr, pf, num_output * sizeof(__fp16));
            memcpy(ptr + num_output, pr, num_output * sizeof(__fp16));
        }
    }

    if (top_blobs.size() == 2)
    {
        cast_float32_to_float16(hidden, top_blobs[1], opt);
    }

    return 0;
}

#endif

} // namespace ncnn
