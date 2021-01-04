// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "lstm_arm.h"

#include <math.h>

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "neon_mathfun_fp16s.h"
#endif
#include "neon_activation.h"
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

LSTM_arm::LSTM_arm()
{
    one_blob_only = false;
    support_inplace = false;
}
int LSTM_arm::create_pipeline(const Option& opt)
{
#if __ARM_NEON
    if (opt.use_fp16_storage)
    {
        ncnn::cast_float32_to_float16(weight_xc_data, weight_xc_data_fp16, opt);
        ncnn::cast_float32_to_float16(weight_hc_data, weight_hc_data_fp16, opt);
    }
#endif // __ARM_NEON

    return 0;
}

#if __ARM_NEON
static int lstm(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& hidden_state, Mat& cell_state, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;

    // 4 x num_output
    Mat gates(num_output, 4, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    // unroll
    for (int t = 0; t < T; t++)
    {
        // clip hidden by continuation indicator
        // h_cont_{t-1} = cont_t * h_{t-1}
        // h_cont_{t-1} = h_{t-1} if cont_t == 1
        //                0       otherwise
        // calculate hidden
        // gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c

        int ti = reverse ? T - 1 - t : t;
        for (int q = 0; q < num_output; q++)
        {
            const float* x = bottom_blob.row(ti);
            const float* hidden_ptr_r = hidden_state;
            const float* bias_c_I = bias_c.row(0);
            const float* bias_c_F = bias_c.row(1);
            const float* bias_c_O = bias_c.row(2);
            const float* bias_c_G = bias_c.row(3);

            float* gates_data_I = gates.row(0);
            float* gates_data_F = gates.row(1);
            float* gates_data_O = gates.row(2);
            float* gates_data_G = gates.row(3);
            // gate I F O G
            const float* weight_xc_I = weight_xc.row(num_output * 0 + q);
            const float* weight_xc_F = weight_xc.row(num_output * 1 + q);
            const float* weight_xc_O = weight_xc.row(num_output * 2 + q);
            const float* weight_xc_G = weight_xc.row(num_output * 3 + q);

            const float* weight_hc_I = weight_hc.row(num_output * 0 + q);
            const float* weight_hc_F = weight_hc.row(num_output * 1 + q);
            const float* weight_hc_O = weight_hc.row(num_output * 2 + q);
            const float* weight_hc_G = weight_hc.row(num_output * 3 + q);

            // float I = bias_c_I[q];
            // float F = bias_c_F[q];
            // float O = bias_c_O[q];
            // float G = bias_c_G[q];
            float32x4_t _sumI = vdupq_n_f32(0.0f);
            float32x4_t _sumF = vdupq_n_f32(0.0f);
            float32x4_t _sumO = vdupq_n_f32(0.0f);
            float32x4_t _sumG = vdupq_n_f32(0.0f);
            int nn_num_size = size >> 2;
            int remain_size = size & 3;
            for (; nn_num_size > 0; nn_num_size--)
            {
                float32x4_t xi = vld1q_f32(x);
                _sumI = vmlaq_f32(_sumI, vld1q_f32(weight_xc_I), xi);
                _sumF = vmlaq_f32(_sumF, vld1q_f32(weight_xc_F), xi);
                _sumO = vmlaq_f32(_sumO, vld1q_f32(weight_xc_O), xi);
                _sumG = vmlaq_f32(_sumG, vld1q_f32(weight_xc_G), xi);
                x += 4;
                weight_xc_I += 4;
                weight_xc_F += 4;
                weight_xc_O += 4;
                weight_xc_G += 4;
            }
            int nn_num_output = num_output >> 2;
            int remain_num_output = num_output & 3;
            for (; nn_num_output > 0; nn_num_output--)
            {
                float32x4_t h_cont = vld1q_f32(hidden_ptr_r);

                _sumI = vmlaq_f32(_sumI, vld1q_f32(weight_hc_I), h_cont);
                _sumF = vmlaq_f32(_sumF, vld1q_f32(weight_hc_F), h_cont);
                _sumO = vmlaq_f32(_sumO, vld1q_f32(weight_hc_O), h_cont);
                _sumG = vmlaq_f32(_sumG, vld1q_f32(weight_hc_G), h_cont);
                hidden_ptr_r += 4;
                weight_hc_I += 4;
                weight_hc_F += 4;
                weight_hc_O += 4;
                weight_hc_G += 4;
            }
            float32x2_t _sum0ss = vadd_f32(vget_low_f32(_sumI), vget_high_f32(_sumI));
            float32x2_t _sum1ss = vadd_f32(vget_low_f32(_sumF), vget_high_f32(_sumF));
            float32x2_t _sum2ss = vadd_f32(vget_low_f32(_sumO), vget_high_f32(_sumO));
            float32x2_t _sum3ss = vadd_f32(vget_low_f32(_sumG), vget_high_f32(_sumG));

            float32x2_t _sum01ss = vpadd_f32(_sum0ss, _sum1ss);
            float32x2_t _sum23ss = vpadd_f32(_sum2ss, _sum3ss);

            float sums0 = vget_lane_f32(_sum01ss, 0) + bias_c_I[q];
            float sums1 = vget_lane_f32(_sum01ss, 1) + bias_c_F[q];
            float sums2 = vget_lane_f32(_sum23ss, 0) + bias_c_O[q];
            float sums3 = vget_lane_f32(_sum23ss, 1) + bias_c_G[q];

            for (; remain_size > 0; remain_size--)
            {
                float xi = *x;
                sums0 += *weight_xc_I * xi;
                sums1 += *weight_xc_F * xi;
                sums2 += *weight_xc_O * xi;
                sums3 += *weight_xc_G * xi;
                x++;
                weight_xc_I++;
                weight_xc_F++;
                weight_xc_O++;
                weight_xc_G++;
            }

            for (; remain_num_output > 0; remain_num_output--)
            {
                float h_cont = *hidden_ptr_r;
                sums0 += *weight_hc_I * h_cont;
                sums1 += *weight_hc_F * h_cont;
                sums2 += *weight_hc_O * h_cont;
                sums3 += *weight_hc_G * h_cont;
                hidden_ptr_r++;
                weight_hc_I++;
                weight_hc_F++;
                weight_hc_O++;
                weight_hc_G++;
            }
            gates_data_I[q] = sums0;
            gates_data_F[q] = sums1;
            gates_data_O[q] = sums2;
            gates_data_G[q] = sums3;
        }

        // lstm unit
        // sigmoid(I)
        // sigmoid(F)
        // sigmoid(O)
        // tanh(G)
        // c_t := f_t .* c_{t-1} + i_t .* g_t
        // h_t := o_t .* tanh[c_t]
        float* output_data = top_blob.row(ti);
        float* cell_ptr = cell_state;
        float* hidden_ptr = hidden_state;
        const float* gates_data_I = gates.row(0);
        const float* gates_data_F = gates.row(1);
        const float* gates_data_O = gates.row(2);
        const float* gates_data_G = gates.row(3);
        int nn_activation = num_output >> 2;
        int remain_activations = num_output & 3;
        for (; nn_activation > 0; nn_activation--)
        {
            float32x4_t I = sigmoid_ps(vld1q_f32(gates_data_I));
            float32x4_t F = sigmoid_ps(vld1q_f32(gates_data_F));
            float32x4_t O = sigmoid_ps(vld1q_f32(gates_data_O));
            float32x4_t G = tanh_ps(vld1q_f32(gates_data_G));
            float32x4_t cell2 = vaddq_f32(vmulq_f32(F, vld1q_f32(cell_ptr)), vmulq_f32(I, G));
            float32x4_t H = vmulq_f32(O, tanh_ps(cell2));
            vst1q_f32(cell_ptr, cell2);
            vst1q_f32(hidden_ptr, H);
            vst1q_f32(output_data, H);
            cell_ptr += 4;
            output_data += 4;
            hidden_ptr += 4;
            gates_data_I += 4;
            gates_data_F += 4;
            gates_data_O += 4;
            gates_data_G += 4;
        }
        for (; remain_activations > 0; remain_activations--)
        {
            float I = *gates_data_I;
            float F = *gates_data_F;
            float O = *gates_data_O;
            float G = *gates_data_G;

            I = 1.f / (1.f + exp(-I));
            F = 1.f / (1.f + exp(-F));
            O = 1.f / (1.f + exp(-O));
            G = tanh(G);
            float cell2 = F * *cell_ptr + I * G;
            float H = O * tanh(cell2);
            *cell_ptr = cell2;
            *hidden_ptr = H;
            *output_data = H;
            cell_ptr++;
            output_data++;
            hidden_ptr++;
            gates_data_I++;
            gates_data_F++;
            gates_data_O++;
            gates_data_G++;
        }

        // no cell output here
    }

    return 0;
}
#if (__ARM_FP & 2)
static int lstm_fp16(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& hidden_state, Mat& cell_state, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;

    // 4 x num_output
    Mat gates(num_output, 4, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    // unroll
    for (int t = 0; t < T; t++)
    {
        // clip hidden by continuation indicator
        // h_cont_{t-1} = cont_t * h_{t-1}
        // h_cont_{t-1} = h_{t-1} if cont_t == 1
        //                0       otherwise
        // calculate hidden
        // gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c

        int ti = reverse ? T - 1 - t : t;
        for (int q = 0; q < num_output; q++)
        {
            const float* x = bottom_blob.row(ti);
            const float* hidden_ptr_r = hidden_state;
            const float* bias_c_I = bias_c.row(0);
            const float* bias_c_F = bias_c.row(1);
            const float* bias_c_O = bias_c.row(2);
            const float* bias_c_G = bias_c.row(3);

            float* gates_data_I = gates.row(0);
            float* gates_data_F = gates.row(1);
            float* gates_data_O = gates.row(2);
            float* gates_data_G = gates.row(3);
            // gate I F O G
            const unsigned short* weight_xc_I = (const unsigned short*)weight_xc.row(num_output * 0 + q);
            const unsigned short* weight_xc_F = (const unsigned short*)weight_xc.row(num_output * 1 + q);
            const unsigned short* weight_xc_O = (const unsigned short*)weight_xc.row(num_output * 2 + q);
            const unsigned short* weight_xc_G = (const unsigned short*)weight_xc.row(num_output * 3 + q);

            const unsigned short* weight_hc_I = (const unsigned short*)weight_hc.row(num_output * 0 + q);
            const unsigned short* weight_hc_F = (const unsigned short*)weight_hc.row(num_output * 1 + q);
            const unsigned short* weight_hc_O = (const unsigned short*)weight_hc.row(num_output * 2 + q);
            const unsigned short* weight_hc_G = (const unsigned short*)weight_hc.row(num_output * 3 + q);

            // float I = bias_c_I[q];
            // float F = bias_c_F[q];
            // float O = bias_c_O[q];
            // float G = bias_c_G[q];
            float32x4_t _sumI = vdupq_n_f32(0.0f);
            float32x4_t _sumF = vdupq_n_f32(0.0f);
            float32x4_t _sumO = vdupq_n_f32(0.0f);
            float32x4_t _sumG = vdupq_n_f32(0.0f);
            int nn_num_size = size >> 2;
            int remain_size = size & 3;
            for (; nn_num_size > 0; nn_num_size--)
            {
                float32x4_t xi = vld1q_f32(x);
                _sumI = vmlaq_f32(_sumI, loadfp16(weight_xc_I), xi);
                _sumF = vmlaq_f32(_sumF, loadfp16(weight_xc_F), xi);
                _sumO = vmlaq_f32(_sumO, loadfp16(weight_xc_O), xi);
                _sumG = vmlaq_f32(_sumG, loadfp16(weight_xc_G), xi);
                x += 4;
                weight_xc_I += 4;
                weight_xc_F += 4;
                weight_xc_O += 4;
                weight_xc_G += 4;
            }
            int nn_num_output = num_output >> 2;
            int remain_num_output = num_output & 3;
            for (; nn_num_output > 0; nn_num_output--)
            {
                float32x4_t h_cont = vld1q_f32(hidden_ptr_r);

                _sumI = vmlaq_f32(_sumI, loadfp16(weight_hc_I), h_cont);
                _sumF = vmlaq_f32(_sumF, loadfp16(weight_hc_F), h_cont);
                _sumO = vmlaq_f32(_sumO, loadfp16(weight_hc_O), h_cont);
                _sumG = vmlaq_f32(_sumG, loadfp16(weight_hc_G), h_cont);
                hidden_ptr_r += 4;
                weight_hc_I += 4;
                weight_hc_F += 4;
                weight_hc_O += 4;
                weight_hc_G += 4;
            }
            if (remain_size)
            {
                unsigned short fp16_weights[4][4] = {{0}};
                float _xi_f[4] = {0};
                // No fast way to convert to fp32 one element at the time
                // so batch an 8 lane vector.
                for (int i = 0; i < remain_size; i++)
                {
                    _xi_f[i] = *x;
                    fp16_weights[0][i] = *weight_xc_I;
                    fp16_weights[1][i] = *weight_xc_F;
                    fp16_weights[2][i] = *weight_xc_O;
                    fp16_weights[3][i] = *weight_xc_G;
                    x++;
                    weight_xc_I++;
                    weight_xc_F++;
                    weight_xc_O++;
                    weight_xc_G++;
                }
                float32x4_t xi = vld1q_f32(_xi_f);
                _sumI = vmlaq_f32(_sumI, loadfp16(fp16_weights[0]), xi);
                _sumF = vmlaq_f32(_sumF, loadfp16(fp16_weights[1]), xi);
                _sumO = vmlaq_f32(_sumO, loadfp16(fp16_weights[2]), xi);
                _sumG = vmlaq_f32(_sumG, loadfp16(fp16_weights[3]), xi);
            }
            if (remain_num_output)
            {
                unsigned short fp16_weights[4][4] = {{0}};
                float _hcont_f[4] = {0};
                // No fast way to convert to fp32 one element at the time
                // so batch an 8 lane vector.
                for (int i = 0; i < remain_num_output; i++)
                {
                    _hcont_f[i] = *hidden_ptr_r;
                    fp16_weights[0][i] = *weight_hc_I;
                    fp16_weights[1][i] = *weight_hc_F;
                    fp16_weights[2][i] = *weight_hc_O;
                    fp16_weights[3][i] = *weight_hc_G;
                    hidden_ptr_r++;
                    weight_hc_I++;
                    weight_hc_F++;
                    weight_hc_O++;
                    weight_hc_G++;
                }
                float32x4_t h_cont = vld1q_f32(_hcont_f);
                _sumI = vmlaq_f32(_sumI, loadfp16(fp16_weights[0]), h_cont);
                _sumF = vmlaq_f32(_sumF, loadfp16(fp16_weights[1]), h_cont);
                _sumO = vmlaq_f32(_sumO, loadfp16(fp16_weights[2]), h_cont);
                _sumG = vmlaq_f32(_sumG, loadfp16(fp16_weights[3]), h_cont);
            }
            float32x2_t _sum0ss = vadd_f32(vget_low_f32(_sumI), vget_high_f32(_sumI));
            float32x2_t _sum1ss = vadd_f32(vget_low_f32(_sumF), vget_high_f32(_sumF));
            float32x2_t _sum2ss = vadd_f32(vget_low_f32(_sumO), vget_high_f32(_sumO));
            float32x2_t _sum3ss = vadd_f32(vget_low_f32(_sumG), vget_high_f32(_sumG));

            float32x2_t _sum01ss = vpadd_f32(_sum0ss, _sum1ss);
            float32x2_t _sum23ss = vpadd_f32(_sum2ss, _sum3ss);

            float sums0 = vget_lane_f32(_sum01ss, 0) + bias_c_I[q];
            float sums1 = vget_lane_f32(_sum01ss, 1) + bias_c_F[q];
            float sums2 = vget_lane_f32(_sum23ss, 0) + bias_c_O[q];
            float sums3 = vget_lane_f32(_sum23ss, 1) + bias_c_G[q];

            gates_data_I[q] = sums0;
            gates_data_F[q] = sums1;
            gates_data_O[q] = sums2;
            gates_data_G[q] = sums3;
        }

        // lstm unit
        // sigmoid(I)
        // sigmoid(F)
        // sigmoid(O)
        // tanh(G)
        // c_t := f_t .* c_{t-1} + i_t .* g_t
        // h_t := o_t .* tanh[c_t]
        float* output_data = top_blob.row(ti);
        float* cell_ptr = cell_state;
        float* hidden_ptr = hidden_state;
        const float* gates_data_I = gates.row(0);
        const float* gates_data_F = gates.row(1);
        const float* gates_data_O = gates.row(2);
        const float* gates_data_G = gates.row(3);
        int nn_activation = num_output >> 2;
        int remain_activations = num_output & 3;
        for (; nn_activation > 0; nn_activation--)
        {
            float32x4_t I = sigmoid_ps(vld1q_f32(gates_data_I));
            float32x4_t F = sigmoid_ps(vld1q_f32(gates_data_F));
            float32x4_t O = sigmoid_ps(vld1q_f32(gates_data_O));
            float32x4_t G = tanh_ps(vld1q_f32(gates_data_G));
            float32x4_t cell2 = vaddq_f32(vmulq_f32(F, vld1q_f32(cell_ptr)), vmulq_f32(I, G));
            float32x4_t H = vmulq_f32(O, tanh_ps(cell2));
            vst1q_f32(cell_ptr, cell2);
            vst1q_f32(hidden_ptr, H);
            vst1q_f32(output_data, H);
            cell_ptr += 4;
            output_data += 4;
            hidden_ptr += 4;
            gates_data_I += 4;
            gates_data_F += 4;
            gates_data_O += 4;
            gates_data_G += 4;
        }
        for (; remain_activations > 0; remain_activations--)
        {
            float I = *gates_data_I;
            float F = *gates_data_F;
            float O = *gates_data_O;
            float G = *gates_data_G;

            I = 1.f / (1.f + exp(-I));
            F = 1.f / (1.f + exp(-F));
            O = 1.f / (1.f + exp(-O));
            G = tanh(G);
            float cell2 = F * *cell_ptr + I * G;
            float H = O * tanh(cell2);
            *cell_ptr = cell2;
            *hidden_ptr = H;
            *output_data = H;
            cell_ptr++;
            output_data++;
            hidden_ptr++;
            gates_data_I++;
            gates_data_F++;
            gates_data_O++;
            gates_data_G++;
        }

        // no cell output here
    }

    return 0;
}
#endif
#endif
int LSTM_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if __ARM_NEON
    int T = bottom_blob.h;
    int num_directions = direction == 2 ? 2 : 1;

    // initial hidden state
    Mat hidden(num_output, 4u, opt.workspace_allocator);
    if (hidden.empty())
        return -100;
    hidden.fill(0.f);
    // internal cell state
    Mat cell(num_output, 4u, opt.workspace_allocator);
    if (cell.empty())
        return -100;
    cell.fill(0.f);

    top_blob.create(num_output * num_directions, T, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
#if (__ARM_FP & 2)
        if (opt.use_fp16_storage && cpu_support_arm_vfpv4())
        {
            // Uni directional
            return lstm_fp16(bottom_blob, top_blob, direction, weight_xc_data_fp16.channel(0), bias_c_data.channel(0), weight_hc_data_fp16.channel(0), hidden, cell, opt);
        }
#endif
        // Uni directional
        return lstm(bottom_blob, top_blob, direction, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden, cell, opt);
    }

    if (direction == 2)
    {
        Mat top_blob_forward(num_output, T, 4u, opt.workspace_allocator);
        if (top_blob_forward.empty())
            return -100;

        Mat top_blob_reverse(num_output, T, 4u, opt.workspace_allocator);
        if (top_blob_reverse.empty())
            return -100;
#if (__ARM_FP & 2)
        if (opt.use_fp16_storage && cpu_support_arm_vfpv4())
        {
            // Uni directional
            int ret0 = lstm_fp16(bottom_blob, top_blob_forward, 0, weight_xc_data_fp16.channel(0), bias_c_data.channel(0), weight_hc_data_fp16.channel(0), hidden, cell, opt);
            if (ret0 != 0)
                return ret0;
            hidden.fill(0.0f);
            cell.fill(0.0f);
            // Uni directional
            int ret1 = lstm_fp16(bottom_blob, top_blob_reverse, 1, weight_xc_data_fp16.channel(1), bias_c_data.channel(1), weight_hc_data_fp16.channel(1), hidden, cell, opt);
            if (ret1 != 0)
                return ret1;
        }
        else
        {
#endif
            // Uni directional
            int ret0 = lstm(bottom_blob, top_blob_forward, 0, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden, cell, opt);
            if (ret0 != 0)
                return ret0;

            hidden.fill(0.0f);
            cell.fill(0.0f);

            // Uni directional
            int ret1 = lstm(bottom_blob, top_blob_reverse, 1, weight_xc_data.channel(1), bias_c_data.channel(1), weight_hc_data.channel(1), hidden, cell, opt);
            if (ret1 != 0)
                return ret1;
#if (__ARM_FP & 2)
        }
#endif

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
#else
    return LSTM::forward(bottom_blob, top_blob, opt);
#endif
}

int LSTM_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if __ARM_NEON
    if (bottom_blobs.size() != 3 || top_blobs.size() != 3)
    {
        return forward(bottom_blobs[0], top_blobs[0], opt);
    }
    const Mat& bottom_blob = bottom_blobs[0];

    int T = bottom_blob.h;
    Mat& top_blob = top_blobs[0];
    Mat& hidden_state = top_blobs[1];
    Mat& cell_state = top_blobs[2];

    //Copy previous states
    hidden_state = bottom_blobs[1].clone(opt.blob_allocator);
    cell_state = bottom_blobs[2].clone(opt.blob_allocator);

    top_blob.create(num_output, T, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;
#if (__ARM_FP & 2)
    if (opt.use_fp16_storage && cpu_support_arm_vfpv4())
    {
        // Uni directional
        return lstm_fp16(bottom_blob, top_blob, direction, weight_xc_data_fp16.channel(0), bias_c_data.channel(0), weight_hc_data_fp16.channel(0), hidden_state, cell_state, opt);
    }
#endif
    // Uni directional
    return lstm(bottom_blob, top_blob, direction, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden_state, cell_state, opt);
#else
    return LSTM::forward(bottom_blobs, top_blobs, opt);
#endif
}

} // namespace ncnn
