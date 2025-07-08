// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_onnx.h"
#include "ir.h"

#include "onnx-ml.pb.h"

namespace pnnx {

namespace onnx2pnnx {

class AdaptiveAvgPool2d : public FuseFunctionPass
{
public:
    const char* match_type_str() const
    {
        return "nn.AdaptiveAvgPool2d";
    }

    const char* type_str() const
    {
        return "nn.AdaptiveAvgPool2d";
    }

    void write(Operator* op, const OnnxFunctionProxy& function) const
    {
        const std::vector<int>& out_shape = op->outputs[0]->shape;

        if (out_shape.size() == 3)
            op->params["output_size"] = std::vector<int> {out_shape[1], out_shape[2]};
        else // if (out_shape.size() == 4)
            op->params["output_size"] = std::vector<int> {out_shape[2], out_shape[3]};
    }
};

REGISTER_GLOBAL_PNNX_FUSE_FUNCTION_PASS(AdaptiveAvgPool2d)

} // namespace onnx2pnnx

} // namespace pnnx
