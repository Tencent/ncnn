// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_onnx.h"
#include "ir.h"

#include "onnx-ml.pb.h"

namespace pnnx {

namespace onnx2pnnx {

class Conv2d : public FuseFunctionPass
{
public:
    const char* match_type_str() const
    {
        return "nn.Conv2d";
    }

    const char* type_str() const
    {
        return "nn.Conv2d";
    }

    void write(Operator* op, const OnnxFunctionProxy& function) const
    {
        const OnnxNodeProxy aten_convolution_onnx = function.typed_node("_aten_convolution_onnx");

        std::vector<int64_t> dilations = aten_convolution_onnx.attribute("dilations");
        std::vector<int64_t> strides = aten_convolution_onnx.attribute("strides");
        std::vector<int64_t> pads = aten_convolution_onnx.attribute("pads");
        int64_t groups = aten_convolution_onnx.attribute("groups");

        const onnx::TensorProto& weight = function.initializer("weight");

        if (pads.size() == 4)
        {
            pads = {pads[0], pads[1]};
        }

        op->params["in_channels"] = weight.dims(1) * groups;
        op->params["out_channels"] = weight.dims(0);
        op->params["kernel_size"] = {weight.dims(2), weight.dims(3)};
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

REGISTER_GLOBAL_PNNX_FUSE_FUNCTION_PASS(Conv2d)

} // namespace onnx2pnnx

} // namespace pnnx
