// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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
