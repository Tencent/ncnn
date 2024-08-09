// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "pass_onnx.h"
#include "ir.h"

#include "onnx-ml.pb.h"

namespace pnnx {

namespace onnx2pnnx {

class AvgPool3d : public FuseFunctionPass
{
public:
    const char* match_type_str() const
    {
        return "nn.AvgPool3d";
    }

    const char* type_str() const
    {
        return "nn.AvgPool3d";
    }

    void write(Operator* op, const OnnxFunctionProxy& function) const
    {
        const OnnxNodeProxy averagepool = function.typed_node("AveragePool");

        std::vector<int64_t> kernel_shape = averagepool.attribute("kernel_shape");
        std::vector<int64_t> strides = averagepool.attribute("strides");
        std::vector<int64_t> pads = averagepool.attribute("pads");
        int64_t ceil_mode = averagepool.attribute("ceil_mode");
        int64_t count_include_pad = averagepool.attribute("count_include_pad");

        if (pads.size() == 6)
        {
            pads = {pads[0], pads[1], pads[2]};
        }

        op->params["kernel_size"] = kernel_shape;
        op->params["stride"] = strides;
        op->params["padding"] = pads;
        op->params["ceil_mode"] = (ceil_mode != 0);
        op->params["count_include_pad"] = (count_include_pad != 0);
        op->params["divisor_override"] = Parameter();
    }
};

REGISTER_GLOBAL_PNNX_FUSE_FUNCTION_PASS(AvgPool3d)

} // namespace onnx2pnnx

} // namespace pnnx
