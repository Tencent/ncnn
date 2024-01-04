// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "convolution1d_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif
#endif // __SSE2__
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#include "convolution1d_packed.h"

Convolution1D_x86::Convolution1D_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int Convolution1D_x86::create_pipeline(const Option& /*opt*/)
{
    if (dynamic_weight)
        return 0;

    int num_input = weight_data_size / kernel_w / num_output;

    convolution1d_transform_kernel_packed(weight_data, weight_data_tm, num_input, num_output, kernel_w);

    weight_data.release();

    return 0;
}

int Convolution1D_x86::destroy_pipeline(const Option& /*opt*/)
{
    return 0;
}

int Convolution1D_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;

    int out_elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX512F__
        out_elempack = num_output % 16 == 0 ? 16 : num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#elif __AVX__
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__
    size_t out_elemsize = elemsize / elempack * out_elempack;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = num_output / out_elempack;

    top_blob.create(outw, outh, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    convolution1d_packed(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, dilation_w, stride_w, activation_type, activation_params, opt);

    return 0;
}

int Convolution1D_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& _weight_data = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    const int _kernel_w = _weight_data.w;
    const int _num_output = _weight_data.c * _weight_data.elempack;

    Mat weight_data_flattened;
    flatten(_weight_data, weight_data_flattened, opt);
    if (weight_data_flattened.empty())
        return -100;

    // weight_data_flattened as pack1
    weight_data_flattened.w *= weight_data_flattened.elempack;
    weight_data_flattened.elemsize /= weight_data_flattened.elempack;
    weight_data_flattened.elempack = 1;

    Mat bias_data_flattened;
    if (bias_term)
    {
        const Mat& _bias_data = bottom_blobs[2];
        flatten(_bias_data, bias_data_flattened, opt);
        if (bias_data_flattened.empty())
            return -100;

        // bias_data_flattened as pack1
        bias_data_flattened.w *= bias_data_flattened.elempack;
        bias_data_flattened.elemsize /= bias_data_flattened.elempack;
        bias_data_flattened.elempack = 1;
    }

    ncnn::Layer* op = ncnn::create_layer_cpu(ncnn::LayerType::Convolution1D);

    ncnn::ParamDict pd;
    pd.set(0, _num_output);
    pd.set(1, _kernel_w);
    pd.set(2, dilation_w);
    pd.set(3, stride_w);
    pd.set(4, pad_left);
    pd.set(15, pad_right);
    pd.set(18, pad_value);
    pd.set(5, bias_term);
    pd.set(6, weight_data_flattened.w);
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    op->load_param(pd);

    ncnn::Mat weights[2];
    weights[0] = weight_data_flattened;
    weights[1] = bias_data_flattened;

    op->load_model(ncnn::ModelBinFromMatArray(weights));

    op->create_pipeline(opt);

    op->forward(bottom_blob, top_blob, opt);

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

} // namespace ncnn
