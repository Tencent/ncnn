// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef FUSED_ACTIVATION_H
#define FUSED_ACTIVATION_H

#include "mat.h"
#include "layer_type.h"

static NCNN_FORCEINLINE float activation_ss(float v, int activation_type, const ncnn::Mat& activation_params)
{
    switch (activation_type)
    {
    case 1:
    {
        v = fmaxf(v, 0.f);
        break;
    }
    case 2:
    {
        float slope = activation_params[0];
        v = v > 0.f ? v : v * slope;
        break;
    }
    case 3:
    {
        float min = activation_params[0];
        float max = activation_params[1];
        if (v < min)
            v = min;
        if (v > max)
            v = max;
        break;
    }
    case 4:
    {
        v = std::min(v, 88.3762626647949f);
        v = std::max(v, -88.3762626647949f);
        v = 1.f / (1.f + expf(-v));
        break;
    }
    case 5:
    {
        v = v * tanhf(logf(expf(v) + 1.f));
        break;
    }
    case 6:
    {
        float alpha = activation_params[0];
        float beta = activation_params[1];
        float lower = -beta / alpha;
        float upper = (1.f / alpha) + lower;
        if (v < lower)
            v = 0.f;
        else if (v > upper)
            ;
        else
            v = v * (v * alpha + beta);
        break;
    }
    }

    return v;
}

static ncnn::Layer* create_activation_layer(int activation_type, const ncnn::Mat& activation_params, const ncnn::Option& opt)
{
    ncnn::Layer* activation = 0;

    if (activation_type == 1)
    {
        activation = ncnn::create_layer_cpu(ncnn::LayerType::ReLU);

        ncnn::ParamDict pd;
        activation->load_param(pd);
    }
    else if (activation_type == 2)
    {
        activation = ncnn::create_layer_cpu(ncnn::LayerType::ReLU);

        ncnn::ParamDict pd;
        pd.set(0, activation_params[0]); // slope
        activation->load_param(pd);
    }
    else if (activation_type == 3)
    {
        activation = ncnn::create_layer_cpu(ncnn::LayerType::Clip);

        ncnn::ParamDict pd;
        pd.set(0, activation_params[0]); // min
        pd.set(1, activation_params[1]); // max

        activation->load_param(pd);
    }
    else if (activation_type == 4)
    {
        activation = ncnn::create_layer_cpu(ncnn::LayerType::Sigmoid);

        ncnn::ParamDict pd;
        activation->load_param(pd);
    }
    else if (activation_type == 5)
    {
        activation = ncnn::create_layer_cpu(ncnn::LayerType::Mish);

        ncnn::ParamDict pd;
        activation->load_param(pd);
    }
    else if (activation_type == 6)
    {
        activation = ncnn::create_layer_cpu(ncnn::LayerType::HardSwish);

        ncnn::ParamDict pd;
        pd.set(0, activation_params[0]); // alpha
        pd.set(1, activation_params[1]); // beta

        activation->load_param(pd);
    }

    if (activation)
    {
        activation->create_pipeline(opt);
    }

    return activation;
}

#endif // FUSED_ACTIVATION_H
