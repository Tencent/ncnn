// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lstm.h"

namespace ncnn {

LSTM::LSTM()
{
    one_blob_only = false;
    support_inplace = false;
}

int LSTM::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    weight_data_size = pd.get(1, 0);
    direction = pd.get(2, 0);
    hidden_size = pd.get(3, num_output);
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

int LSTM::load_model(const ModelBin& mb)
{
    int num_directions = direction == 2 ? 2 : 1;

    int size = weight_data_size / num_directions / hidden_size / 4;

    // raw weight data
    weight_xc_data = mb.load(size, hidden_size * 4, num_directions, 0);
    if (weight_xc_data.empty())
        return -100;

    bias_c_data = mb.load(hidden_size, 4, num_directions, 0);
    if (bias_c_data.empty())
        return -100;

    weight_hc_data = mb.load(num_output, hidden_size * 4, num_directions, 0);
    if (weight_hc_data.empty())
        return -100;

    if (num_output != hidden_size)
    {
        weight_hr_data = mb.load(hidden_size, num_output, num_directions, 0);
        if (weight_hr_data.empty())
            return -100;
    }

#if NCNN_INT8
    if (int8_scale_term)
    {
        weight_xc_data_int8_scales = mb.load(hidden_size * 4, num_directions, 1);
        weight_hc_data_int8_scales = mb.load(hidden_size * 4, num_directions, 1);
    }
#endif // NCNN_INT8

    return 0;
}

static int lstm(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;
    int hidden_size = cell_state.w;

    // 4 x hidden_size
    Mat gates(4, hidden_size, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    Mat tmp_hidden_state;
    if (num_output != hidden_size)
    {
        tmp_hidden_state.create(hidden_size, 4u, opt.workspace_allocator);
        if (tmp_hidden_state.empty())
            return -100;
    }

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

        const float* x = bottom_blob.row(ti);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < hidden_size; q++)
        {
            const float* bias_c_I = bias_c.row(0);
            const float* bias_c_F = bias_c.row(1);
            const float* bias_c_O = bias_c.row(2);
            const float* bias_c_G = bias_c.row(3);

            float* gates_data = gates.row(q);

            // gate I F O G
            const float* weight_xc_I = weight_xc.row(hidden_size * 0 + q);
            const float* weight_xc_F = weight_xc.row(hidden_size * 1 + q);
            const float* weight_xc_O = weight_xc.row(hidden_size * 2 + q);
            const float* weight_xc_G = weight_xc.row(hidden_size * 3 + q);

            const float* weight_hc_I = weight_hc.row(hidden_size * 0 + q);
            const float* weight_hc_F = weight_hc.row(hidden_size * 1 + q);
            const float* weight_hc_O = weight_hc.row(hidden_size * 2 + q);
            const float* weight_hc_G = weight_hc.row(hidden_size * 3 + q);

            float I = bias_c_I[q];
            float F = bias_c_F[q];
            float O = bias_c_O[q];
            float G = bias_c_G[q];

            for (int i = 0; i < size; i++)
            {
                float xi = x[i];

                I += weight_xc_I[i] * xi;
                F += weight_xc_F[i] * xi;
                O += weight_xc_O[i] * xi;
                G += weight_xc_G[i] * xi;
            }

            for (int i = 0; i < num_output; i++)
            {
                float h_cont = hidden_state[i];

                I += weight_hc_I[i] * h_cont;
                F += weight_hc_F[i] * h_cont;
                O += weight_hc_O[i] * h_cont;
                G += weight_hc_G[i] * h_cont;
            }

            gates_data[0] = I;
            gates_data[1] = F;
            gates_data[2] = O;
            gates_data[3] = G;
        }

        // lstm unit
        // sigmoid(I)
        // sigmoid(F)
        // sigmoid(O)
        // tanh(G)
        // c_t := f_t .* c_{t-1} + i_t .* g_t
        // h_t := o_t .* tanh[c_t]
        float* output_data = top_blob.row(ti);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < hidden_size; q++)
        {
            const float* gates_data = gates.row(q);

            float I = gates_data[0];
            float F = gates_data[1];
            float O = gates_data[2];
            float G = gates_data[3];

            I = 1.f / (1.f + expf(-I));
            F = 1.f / (1.f + expf(-F));
            O = 1.f / (1.f + expf(-O));
            G = tanhf(G);

            float cell2 = F * cell_state[q] + I * G;
            float H = O * tanhf(cell2);
            cell_state[q] = cell2;

            if (num_output == hidden_size)
            {
                hidden_state[q] = H;
                output_data[q] = H;
            }
            else
            {
                tmp_hidden_state[q] = H;
            }
        }

        if (num_output != hidden_size)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < num_output; q++)
            {
                const float* hr = weight_hr.row(q);

                float H = 0;
                for (int i = 0; i < hidden_size; i++)
                {
                    H += tmp_hidden_state[i] * hr[i];
                }

                hidden_state[q] = H;
                output_data[q] = H;
            }
        }
    }

    return 0;
}

#if NCNN_INT8
static int lstm_int8(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc_int8, const float* weight_xc_int8_scales, const Mat& bias_c, const Mat& weight_hc_int8, const float* weight_hc_int8_scales, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;
    int hidden_size = cell_state.w;

    // 4 x hidden_size
    Mat gates(4, hidden_size, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    Mat tmp_hidden_state;
    if (num_output != hidden_size)
    {
        tmp_hidden_state.create(hidden_size, 4u, opt.workspace_allocator);
        if (tmp_hidden_state.empty())
            return -100;
    }

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
        // clip hidden by continuation indicator
        // h_cont_{t-1} = cont_t * h_{t-1}
        // h_cont_{t-1} = h_{t-1} if cont_t == 1
        //                0       otherwise
        // calculate hidden
        // gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c

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
        for (int q = 0; q < hidden_size; q++)
        {
            const float* bias_c_I = bias_c.row(0);
            const float* bias_c_F = bias_c.row(1);
            const float* bias_c_O = bias_c.row(2);
            const float* bias_c_G = bias_c.row(3);

            float* gates_data = gates.row(q);

            // gate I F O G
            const signed char* weight_xc_int8_I = weight_xc_int8.row<const signed char>(hidden_size * 0 + q);
            const signed char* weight_xc_int8_F = weight_xc_int8.row<const signed char>(hidden_size * 1 + q);
            const signed char* weight_xc_int8_O = weight_xc_int8.row<const signed char>(hidden_size * 2 + q);
            const signed char* weight_xc_int8_G = weight_xc_int8.row<const signed char>(hidden_size * 3 + q);

            const signed char* weight_hc_int8_I = weight_hc_int8.row<const signed char>(hidden_size * 0 + q);
            const signed char* weight_hc_int8_F = weight_hc_int8.row<const signed char>(hidden_size * 1 + q);
            const signed char* weight_hc_int8_O = weight_hc_int8.row<const signed char>(hidden_size * 2 + q);
            const signed char* weight_hc_int8_G = weight_hc_int8.row<const signed char>(hidden_size * 3 + q);

            const float descale_xc_I = 1.f / weight_xc_int8_scales[hidden_size * 0 + q];
            const float descale_xc_F = 1.f / weight_xc_int8_scales[hidden_size * 1 + q];
            const float descale_xc_O = 1.f / weight_xc_int8_scales[hidden_size * 2 + q];
            const float descale_xc_G = 1.f / weight_xc_int8_scales[hidden_size * 3 + q];
            const float descale_hc_I = 1.f / weight_hc_int8_scales[hidden_size * 0 + q];
            const float descale_hc_F = 1.f / weight_hc_int8_scales[hidden_size * 1 + q];
            const float descale_hc_O = 1.f / weight_hc_int8_scales[hidden_size * 2 + q];
            const float descale_hc_G = 1.f / weight_hc_int8_scales[hidden_size * 3 + q];

            int Ix = 0;
            int Fx = 0;
            int Ox = 0;
            int Gx = 0;
            for (int i = 0; i < size; i++)
            {
                signed char xi = x[i];

                Ix += weight_xc_int8_I[i] * xi;
                Fx += weight_xc_int8_F[i] * xi;
                Ox += weight_xc_int8_O[i] * xi;
                Gx += weight_xc_int8_G[i] * xi;
            }

            int Ih = 0;
            int Fh = 0;
            int Oh = 0;
            int Gh = 0;
            for (int i = 0; i < num_output; i++)
            {
                signed char h_cont = hs[i];

                Ih += weight_hc_int8_I[i] * h_cont;
                Fh += weight_hc_int8_F[i] * h_cont;
                Oh += weight_hc_int8_O[i] * h_cont;
                Gh += weight_hc_int8_G[i] * h_cont;
            }

            float I = bias_c_I[q] + Ix * (descale_x * descale_xc_I) + Ih * (descale_h * descale_hc_I);
            float F = bias_c_F[q] + Fx * (descale_x * descale_xc_F) + Fh * (descale_h * descale_hc_F);
            float O = bias_c_O[q] + Ox * (descale_x * descale_xc_O) + Oh * (descale_h * descale_hc_O);
            float G = bias_c_G[q] + Gx * (descale_x * descale_xc_G) + Gh * (descale_h * descale_hc_G);

            gates_data[0] = I;
            gates_data[1] = F;
            gates_data[2] = O;
            gates_data[3] = G;
        }

        // lstm unit
        // sigmoid(I)
        // sigmoid(F)
        // sigmoid(O)
        // tanh(G)
        // c_t := f_t .* c_{t-1} + i_t .* g_t
        // h_t := o_t .* tanh[c_t]
        float* output_data = top_blob.row(ti);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < hidden_size; q++)
        {
            const float* gates_data = gates.row(q);

            float I = gates_data[0];
            float F = gates_data[1];
            float O = gates_data[2];
            float G = gates_data[3];

            I = 1.f / (1.f + expf(-I));
            F = 1.f / (1.f + expf(-F));
            O = 1.f / (1.f + expf(-O));
            G = tanhf(G);

            float cell2 = F * cell_state[q] + I * G;
            float H = O * tanhf(cell2);
            cell_state[q] = cell2;

            if (num_output == hidden_size)
            {
                hidden_state[q] = H;
                output_data[q] = H;
            }
            else
            {
                tmp_hidden_state[q] = H;
            }
        }

        if (num_output != hidden_size)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < num_output; q++)
            {
                const float* hr = weight_hr.row(q);

                float H = 0;
                for (int i = 0; i < hidden_size; i++)
                {
                    H += tmp_hidden_state[i] * hr[i];
                }

                hidden_state[q] = H;
                output_data[q] = H;
            }
        }
    }

    return 0;
}
#endif // NCNN_INT8

int LSTM::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int T = bottom_blob.h;

    int num_directions = direction == 2 ? 2 : 1;

    // initial hidden state
    Mat hidden(num_output, 4u, opt.workspace_allocator);
    if (hidden.empty())
        return -100;
    hidden.fill(0.f);

    Mat cell(hidden_size, 4u, opt.workspace_allocator);
    if (cell.empty())
        return -100;
    cell.fill(0.f);

    top_blob.create(num_output * num_directions, T, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
#if NCNN_INT8
        if (int8_scale_term)
        {
            int ret = lstm_int8(bottom_blob, top_blob, direction, weight_xc_data.channel(0), weight_xc_data_int8_scales.row(0), bias_c_data.channel(0), weight_hc_data.channel(0), weight_hc_data_int8_scales.row(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden, cell, opt);
            if (ret != 0)
                return ret;
        }
        else
#endif
        {
            int ret = lstm(bottom_blob, top_blob, direction, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden, cell, opt);
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
            int ret = lstm_int8(bottom_blob, top_blob_forward, 0, weight_xc_data.channel(0), weight_xc_data_int8_scales.row(0), bias_c_data.channel(0), weight_hc_data.channel(0), weight_hc_data_int8_scales.row(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden, cell, opt);
            if (ret != 0)
                return ret;
        }
        else
#endif
        {
            int ret = lstm(bottom_blob, top_blob_forward, 0, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden, cell, opt);
            if (ret != 0)
                return ret;
        }

        hidden.fill(0.0f);
        cell.fill(0.0f);

#if NCNN_INT8
        if (int8_scale_term)
        {
            int ret = lstm_int8(bottom_blob, top_blob_reverse, 1, weight_xc_data.channel(1), weight_xc_data_int8_scales.row(1), bias_c_data.channel(1), weight_hc_data.channel(1), weight_hc_data_int8_scales.row(1), num_output == hidden_size ? Mat() : weight_hr_data.channel(1), hidden, cell, opt);
            if (ret != 0)
                return ret;
        }
        else
#endif
        {
            int ret = lstm(bottom_blob, top_blob_reverse, 1, weight_xc_data.channel(1), bias_c_data.channel(1), weight_hc_data.channel(1), num_output == hidden_size ? Mat() : weight_hr_data.channel(1), hidden, cell, opt);
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

int LSTM::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int T = bottom_blob.h;
    int num_directions = direction == 2 ? 2 : 1;

    Mat hidden;
    Mat cell;
    Allocator* hidden_cell_allocator = top_blobs.size() == 3 ? opt.blob_allocator : opt.workspace_allocator;
    if (bottom_blobs.size() == 3)
    {
        hidden = bottom_blobs[1].clone(hidden_cell_allocator);
        cell = bottom_blobs[2].clone(hidden_cell_allocator);
    }
    else
    {
        hidden.create(num_output, num_directions, 4u, hidden_cell_allocator);
        if (hidden.empty())
            return -100;
        hidden.fill(0.f);

        cell.create(hidden_size, num_directions, 4u, hidden_cell_allocator);
        if (cell.empty())
            return -100;
        cell.fill(0.f);
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
            int ret = lstm_int8(bottom_blob, top_blob, direction, weight_xc_data.channel(0), weight_xc_data_int8_scales.row(0), bias_c_data.channel(0), weight_hc_data.channel(0), weight_hc_data_int8_scales.row(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden, cell, opt);
            if (ret != 0)
                return ret;
        }
        else
#endif
        {
            int ret = lstm(bottom_blob, top_blob, direction, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden, cell, opt);
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
        Mat cell0 = cell.row_range(0, 1);
#if NCNN_INT8
        if (int8_scale_term)
        {
            int ret = lstm_int8(bottom_blob, top_blob_forward, 0, weight_xc_data.channel(0), weight_xc_data_int8_scales.row(0), bias_c_data.channel(0), weight_hc_data.channel(0), weight_hc_data_int8_scales.row(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden0, cell0, opt);
            if (ret != 0)
                return ret;
        }
        else
#endif
        {
            int ret = lstm(bottom_blob, top_blob_forward, 0, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden0, cell0, opt);
            if (ret != 0)
                return ret;
        }

        Mat hidden1 = hidden.row_range(1, 1);
        Mat cell1 = cell.row_range(1, 1);
#if NCNN_INT8
        if (int8_scale_term)
        {
            int ret = lstm_int8(bottom_blob, top_blob_reverse, 1, weight_xc_data.channel(1), weight_xc_data_int8_scales.row(1), bias_c_data.channel(1), weight_hc_data.channel(1), weight_hc_data_int8_scales.row(1), num_output == hidden_size ? Mat() : weight_hr_data.channel(1), hidden1, cell1, opt);
            if (ret != 0)
                return ret;
        }
        else
#endif
        {
            int ret = lstm(bottom_blob, top_blob_reverse, 1, weight_xc_data.channel(1), bias_c_data.channel(1), weight_hc_data.channel(1), num_output == hidden_size ? Mat() : weight_hr_data.channel(1), hidden1, cell1, opt);
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

    if (top_blobs.size() == 3)
    {
        top_blobs[1] = hidden;
        top_blobs[2] = cell;
    }

    return 0;
}

} // namespace ncnn
