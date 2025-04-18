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

#include "rnn.h"

namespace ncnn {

RNN::RNN()
{
    one_blob_only = false;
    support_inplace = false;
}

int RNN::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    weight_data_size = pd.get(1, 0);
    direction = pd.get(2, 0);
    int8_scale_term = pd.get(8, 0);

    if (int8_scale_term)
    {
#if !NCNN_INT8
        NCNN_LOGE("please build ncnn with NCNN_INT8 enabled for int8 inference");
        return -1;
#endif
    }

    return 0;
}

int RNN::load_model(const ModelBin& mb)
{
    int num_directions = direction == 2 ? 2 : 1;

    int size = weight_data_size / num_directions / num_output;

    // raw weight data
    weight_xc_data = mb.load(size, num_output, num_directions, 0);
    if (weight_xc_data.empty())
        return -100;

    bias_c_data = mb.load(num_output, 1, num_directions, 0);
    if (bias_c_data.empty())
        return -100;

    weight_hc_data = mb.load(num_output, num_output, num_directions, 0);
    if (weight_hc_data.empty())
        return -100;

#if NCNN_INT8
    if (int8_scale_term)
    {
        weight_xc_data_int8_scales = mb.load(num_output, num_directions, 1);
        weight_hc_data_int8_scales = mb.load(num_output, num_directions, 1);
    }
#endif // NCNN_INT8

    return 0;
}

static int rnn(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& hidden_state, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;

    // num_output
    Mat gates(num_output, 4u, opt.workspace_allocator);
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
            const float* weight_xc_ptr = weight_xc.row(q);
            const float* weight_hc_ptr = weight_hc.row(q);

            float H = bias_c[q];

            for (int i = 0; i < size; i++)
            {
                H += weight_xc_ptr[i] * x[i];
            }

            for (int i = 0; i < num_output; i++)
            {
                H += weight_hc_ptr[i] * hidden_state[i];
            }

            H = tanhf(H);

            gates[q] = H;
        }

        float* output_data = top_blob.row(ti);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_output; q++)
        {
            float H = gates[q];

            hidden_state[q] = H;
            output_data[q] = H;
        }
    }

    return 0;
}

#if NCNN_INT8
static int rnn_int8(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc_int8, const float* weight_xc_int8_scales, const Mat& bias_c, const Mat& weight_hc_int8, const float* weight_hc_int8_scales, Mat& hidden_state, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;

    // num_output
    Mat gates(num_output, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    // dynamic quantize bottom_blob
    Mat bottom_blob_int8(size, T, (size_t)1u, 1, opt.workspace_allocator);
    Mat bottom_blob_int8_scales(T, (size_t)4u, 1, opt.workspace_allocator);
    {
        for (int t = 0; t < T; t++)
        {
            const float* x = bottom_blob.row(t);

            float absmax = 0.f;
            for (int i = 0; i < size; i++)
            {
                absmax = std::max(absmax, (float)fabs(x[i]));
            }

            bottom_blob_int8_scales[t] = 127.f / absmax;
        }

        Option opt_quant = opt;
        opt_quant.blob_allocator = opt.workspace_allocator;
        opt_quant.use_packing_layout = false;
        quantize_to_int8(bottom_blob, bottom_blob_int8, bottom_blob_int8_scales, opt_quant);
    }

    Mat hidden_state_int8(num_output, (size_t)1u, 1, opt.workspace_allocator);
    Mat hidden_state_int8_scales(1, (size_t)4u, 1, opt.workspace_allocator);

    // unroll
    for (int t = 0; t < T; t++)
    {
        int ti = reverse ? T - 1 - t : t;

        // dynamic quantize hidden_state
        {
            float absmax = 0.f;
            for (int i = 0; i < num_output; i++)
            {
                absmax = std::max(absmax, (float)fabs(hidden_state[i]));
            }

            if (absmax == 0.f)
            {
                hidden_state_int8_scales[0] = 1.f;
                hidden_state_int8.fill<signed char>(0);
            }
            else
            {
                hidden_state_int8_scales[0] = 127.f / absmax;

                Option opt_quant = opt;
                opt_quant.blob_allocator = opt.workspace_allocator;
                opt_quant.use_packing_layout = false;
                quantize_to_int8(hidden_state, hidden_state_int8, hidden_state_int8_scales, opt_quant);
            }
        }

        const signed char* x = bottom_blob_int8.row<const signed char>(ti);
        const signed char* hs = hidden_state_int8;
        const float descale_x = 1.f / bottom_blob_int8_scales[ti];
        const float descale_h = 1.f / hidden_state_int8_scales[0];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_output; q++)
        {
            const signed char* weight_xc_int8_ptr = weight_xc_int8.row<const signed char>(q);
            const signed char* weight_hc_int8_ptr = weight_hc_int8.row<const signed char>(q);

            const float descale_xc = 1.f / weight_xc_int8_scales[q];
            const float descale_hc = 1.f / weight_hc_int8_scales[q];

            int Hx = 0;
            for (int i = 0; i < size; i++)
            {
                Hx += weight_xc_int8_ptr[i] * x[i];
            }

            int Hh = 0;
            for (int i = 0; i < num_output; i++)
            {
                Hh += weight_hc_int8_ptr[i] * hs[i];
            }

            float H = bias_c[q] + Hx * (descale_x * descale_xc) + Hh * (descale_h * descale_hc);

            H = tanhf(H);

            gates[q] = H;
        }

        float* output_data = top_blob.row(ti);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_output; q++)
        {
            float H = gates[q];

            hidden_state[q] = H;
            output_data[q] = H;
        }
    }

    return 0;
}
#endif // NCNN_INT8

int RNN::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
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
#if NCNN_INT8
        if (int8_scale_term)
        {
            int ret = rnn_int8(bottom_blob, top_blob, direction, weight_xc_data.channel(0), weight_xc_data_int8_scales.row(0), bias_c_data.channel(0), weight_hc_data.channel(0), weight_hc_data_int8_scales.row(0), hidden, opt);
            if (ret != 0)
                return ret;
        }
        else
#endif
        {
            int ret = rnn(bottom_blob, top_blob, direction, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden, opt);
            if (ret != 0)
                return ret;
        }
    }

    if (direction == 2)
    {
        Mat top_blob_forward(num_output, T, 4u, opt.workspace_allocator);
        if (top_blob_forward.empty())
            return -100;

        Mat top_blob_reverse(num_output, T, 4u, opt.workspace_allocator);
        if (top_blob_reverse.empty())
            return -100;

#if NCNN_INT8
        if (int8_scale_term)
        {
            int ret = rnn_int8(bottom_blob, top_blob_forward, 0, weight_xc_data.channel(0), weight_xc_data_int8_scales.row(0), bias_c_data.channel(0), weight_hc_data.channel(0), weight_hc_data_int8_scales.row(0), hidden, opt);
            if (ret != 0)
                return ret;
        }
        else
#endif
        {
            int ret = rnn(bottom_blob, top_blob_forward, 0, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden, opt);
            if (ret != 0)
                return ret;
        }

        hidden.fill(0.0f);

#if NCNN_INT8
        if (int8_scale_term)
        {
            int ret = rnn_int8(bottom_blob, top_blob_reverse, 1, weight_xc_data.channel(1), weight_xc_data_int8_scales.row(1), bias_c_data.channel(1), weight_hc_data.channel(1), weight_hc_data_int8_scales.row(1), hidden, opt);
            if (ret != 0)
                return ret;
        }
        else
#endif
        {
            int ret = rnn(bottom_blob, top_blob_reverse, 1, weight_xc_data.channel(1), bias_c_data.channel(1), weight_hc_data.channel(1), hidden, opt);
            if (ret != 0)
                return ret;
        }

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
}

int RNN::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
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
#if NCNN_INT8
        if (int8_scale_term)
        {
            int ret = rnn_int8(bottom_blob, top_blob, direction, weight_xc_data.channel(0), weight_xc_data_int8_scales.row(0), bias_c_data.channel(0), weight_hc_data.channel(0), weight_hc_data_int8_scales.row(0), hidden, opt);
            if (ret != 0)
                return ret;
        }
        else
#endif
        {
            int ret = rnn(bottom_blob, top_blob, direction, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden, opt);
            if (ret != 0)
                return ret;
        }
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
#if NCNN_INT8
        if (int8_scale_term)
        {
            int ret = rnn_int8(bottom_blob, top_blob_forward, 0, weight_xc_data.channel(0), weight_xc_data_int8_scales.row(0), bias_c_data.channel(0), weight_hc_data.channel(0), weight_hc_data_int8_scales.row(0), hidden0, opt);
            if (ret != 0)
                return ret;
        }
        else
#endif
        {
            int ret = rnn(bottom_blob, top_blob_forward, 0, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden0, opt);
            if (ret != 0)
                return ret;
        }

        Mat hidden1 = hidden.row_range(1, 1);
#if NCNN_INT8
        if (int8_scale_term)
        {
            int ret = rnn_int8(bottom_blob, top_blob_reverse, 1, weight_xc_data.channel(1), weight_xc_data_int8_scales.row(1), bias_c_data.channel(1), weight_hc_data.channel(1), weight_hc_data_int8_scales.row(1), hidden1, opt);
            if (ret != 0)
                return ret;
        }
        else
#endif
        {
            int ret = rnn(bottom_blob, top_blob_reverse, 1, weight_xc_data.channel(1), bias_c_data.channel(1), weight_hc_data.channel(1), hidden1, opt);
            if (ret != 0)
                return ret;
        }

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
}

} // namespace ncnn
