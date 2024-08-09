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

class Conv3d : public FuseFunctionPass
{
public:
    const char* match_type_str() const
    {
        return "nn.Conv3d";
    }

    const char* type_str() const
    {
        return "nn.Conv3d";
    }

    void write(Operator* op, const OnnxFunctionProxy& function) const
    {
        const OnnxNodeProxy aten_convolution_onnx = function.typed_node("_aten_convolution_onnx");

        std::vector<int64_t> dilations = aten_convolution_onnx.attribute("dilations");
        std::vector<int64_t> strides = aten_convolution_onnx.attribute("strides");
        std::vector<int64_t> pads = aten_convolution_onnx.attribute("pads");
        int64_t groups = aten_convolution_onnx.attribute("groups");

        const onnx::TensorProto& weight = function.initializer("weight");

        if (pads.size() == 6)
        {
            pads = {pads[0], pads[1], pads[2]};
        }

        op->params["in_channels"] = weight.dims(1) * groups;
        op->params["out_channels"] = weight.dims(0);
        op->params["kernel_size"] = {weight.dims(2), weight.dims(3), weight.dims(4)};
        op->params["dilation"] = dilations;
        op->params["stride"] = strides;
        op->params["padding"] = pads;
        op->params["groups"] = groups;
        op->params["bias"] = function.has_initializer("bias");
        op->params["padding_mode"] = "zeros";

        op->attrs["weight"] = weight;
        if (function.has_initializer("bias"))
        {
            op->attrs["bias"] = function.initializer("bias");
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_FUNCTION_PASS(Conv3d)

} // namespace onnx2pnnx

} // namespace pnnx
