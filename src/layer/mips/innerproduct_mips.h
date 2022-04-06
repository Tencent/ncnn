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

#ifndef LAYER_INNERPRODUCT_MIPS_H
#define LAYER_INNERPRODUCT_MIPS_H

#include "innerproduct.h"

namespace ncnn {

class InnerProduct_mips : virtual public InnerProduct
{
public:
    InnerProduct_mips();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
#if NCNN_INT8
    int create_pipeline_int8_mips(const Option& opt);
    int forward_int8_mips(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif

public:
    Layer* flatten;
    Layer* activation;

#if NCNN_INT8
    // int8
    Mat weight_data_int8;
    Mat scales_in;
#endif
};

} // namespace ncnn

#endif // LAYER_INNERPRODUCT_MIPS_H
