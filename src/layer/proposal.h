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

#ifndef LAYER_PROPOSAL_H
#define LAYER_PROPOSAL_H

#include "layer.h"

namespace ncnn {

class Proposal : public Layer
{
public:
    Proposal();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    // param
    int feat_stride;
    int base_size;
    int pre_nms_topN;
    int after_nms_topN;
    float nms_thresh;
    int min_size;

    Mat ratios;
    Mat scales;

    Mat anchors;
};

} // namespace ncnn

#endif // LAYER_PROPOSAL_H
