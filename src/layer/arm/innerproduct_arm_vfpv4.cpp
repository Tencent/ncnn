// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "innerproduct_arm.h"

#include "cpu.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"
#include "arm_usability.h"

namespace ncnn {

#include "innerproduct_fp16s.h"
#include "innerproduct_gemm_fp16s.h"

int InnerProduct_arm::create_pipeline_fp16s(const Option& opt)
{
    const int num_input = weight_data_size / num_output;

    innerproduct_transform_kernel_fp16s_neon(weight_data, weight_data_tm, num_input, num_output, opt);

#if NCNN_ARM82
    if (ncnn::cpu_support_arm_asimdhp() && opt.use_fp16_arithmetic)
    {
        ncnn::cast_float32_to_float16(bias_data, bias_data_fp16, opt);
    }
#endif

    weight_data.release();

    return 0;
}

int InnerProduct_arm::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int num_input = weight_data_size / num_output;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input)
    {
        // gemm
        int h = bottom_blob.h;
        size_t elemsize = bottom_blob.elemsize;
        int elempack = bottom_blob.elempack;

        top_blob.create(num_output, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        innerproduct_gemm_fp16s_neon(bottom_blob, top_blob, weight_data_tm, bias_data, activation_type, activation_params, opt);

        return 0;
    }

    // flatten
    Mat bottom_blob_flattened = bottom_blob;
    if (bottom_blob.dims != 1)
    {
        Option opt_flatten = opt;
        opt_flatten.blob_allocator = opt.workspace_allocator;

        flatten->forward(bottom_blob, bottom_blob_flattened, opt_flatten);
    }

    size_t elemsize = bottom_blob_flattened.elemsize;
    int elempack = bottom_blob_flattened.elempack;

    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (out_elempack == 4)
    {
        innerproduct_pack4_fp16s_neon(bottom_blob_flattened, top_blob, weight_data_tm, bias_data, activation_type, activation_params, opt);
    }

    if (out_elempack == 1)
    {
        innerproduct_fp16s_neon(bottom_blob_flattened, top_blob, weight_data_tm, bias_data, activation_type, activation_params, opt);
    }

    return 0;
}

} // namespace ncnn
