// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_onnx.h"
#include "ir.h"

#include "onnx-ml.pb.h"

namespace pnnx {

namespace onnx2pnnx {

class GELU : public FuseFunctionPass
{
public:
    const char* match_type_str() const
    {
        return "nn.GELU";
    }

    const char* type_str() const
    {
        return "nn.GELU";
    }

    void write(Operator* op, const OnnxFunctionProxy& function) const
    {
        bool approximate_none = function.has_typed_node("_aten_gelu_approximate_none");
        bool approximate_tanh = function.has_typed_node("_aten_gelu_approximate_tanh");

        if (approximate_none)
            op->params["approximate"] = "none";

        if (approximate_tanh)
            op->params["approximate"] = "tanh";
    }
};

REGISTER_GLOBAL_PNNX_FUSE_FUNCTION_PASS(GELU)

} // namespace onnx2pnnx

} // namespace pnnx
