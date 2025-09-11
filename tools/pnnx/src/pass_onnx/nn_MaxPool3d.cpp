// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_onnx.h"
#include "ir.h"

#include "onnx-ml.pb.h"

namespace pnnx {

namespace onnx2pnnx {

class MaxPool3d : public FuseFunctionPass
{
public:
    const char* match_type_str() const
    {
        return "nn.MaxPool3d";
    }

    const char* type_str() const
    {
        return "nn.MaxPool3d";
    }

    void write(Operator* op, const OnnxFunctionProxy& function) const
    {
        const OnnxNodeProxy aten_max_pool_with_indices_onnx = function.typed_node("_aten_max_pool_with_indices_onnx");

        std::vector<int64_t> kernel_size = aten_max_pool_with_indices_onnx.attribute("kernel_size");
        std::vector<int64_t> dilation = aten_max_pool_with_indices_onnx.attribute("dilation");
        std::vector<int64_t> stride = aten_max_pool_with_indices_onnx.attribute("stride");
        std::vector<int64_t> padding = aten_max_pool_with_indices_onnx.attribute("padding");
        int64_t ceil_mode = aten_max_pool_with_indices_onnx.attribute("ceil_mode");

        if (padding.size() == 6)
        {
            padding = {padding[0], padding[1], padding[2]};
        }

        op->params["kernel_size"] = kernel_size;
        op->params["dilation"] = dilation;
        op->params["stride"] = stride;
        op->params["padding"] = padding;
        op->params["ceil_mode"] = (ceil_mode != 0);
        op->params["return_indices"] = (function.function.output_size() != 1);
    }
};

REGISTER_GLOBAL_PNNX_FUSE_FUNCTION_PASS(MaxPool3d)

} // namespace onnx2pnnx

} // namespace pnnx
