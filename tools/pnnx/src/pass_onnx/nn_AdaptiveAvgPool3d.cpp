// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_onnx.h"
#include "ir.h"

#include "onnx-ml.pb.h"

namespace pnnx {

namespace onnx2pnnx {

class AdaptiveAvgPool3d : public FuseFunctionPass
{
public:
    const char* match_type_str() const
    {
        return "nn.AdaptiveAvgPool3d";
    }

    const char* type_str() const
    {
        return "nn.AdaptiveAvgPool3d";
    }

    void write(Operator* op, const OnnxFunctionProxy& function) const
    {
        const std::vector<int>& out_shape = op->outputs[0]->shape;

        if (out_shape.size() == 4)
            op->params["output_size"] = std::vector<int> {out_shape[1], out_shape[2], out_shape[3]};
        else // if (out_shape.size() == 5)
            op->params["output_size"] = std::vector<int> {out_shape[2], out_shape[3], out_shape[4]};
    }
};

REGISTER_GLOBAL_PNNX_FUSE_FUNCTION_PASS(AdaptiveAvgPool3d)

} // namespace onnx2pnnx

} // namespace pnnx
