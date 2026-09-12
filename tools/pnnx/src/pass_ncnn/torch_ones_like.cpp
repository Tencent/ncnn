// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_ones_like : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.ones_like         op_0        1 1 input out dtype=%dtype
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 4
pnnx.Input              input       0 1 input
BinaryOp                op_0        1 1 input zero_out 0=2 1=1 2=0.0
BinaryOp                op_1        1 1 zero_out out 0=0 1=1 2=1.0
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators,
               const std::map<std::string, Parameter>& captured_params,
               const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        // the BinaryOp lowering produces storage in the input's arithmetic
        // type; only apply it when the output dtype matches (no explicit
        // dtype override, or torch.float), otherwise keep torch.ones_like
        const std::map<std::string, Parameter>::const_iterator it = captured_params.find("dtype");
        bool dtype_ok = false;
        if (it == captured_params.end())
            dtype_ok = true;
        else
        {
            const Parameter& dt = it->second;
            if (dt.type == 0)
                dtype_ok = true; // dtype=None: inherits the input dtype
            if (dt.type == 2 && dt.i == 6)
                dtype_ok = true; // torch.float
            if (dt.type == 4 && dt.s == "torch.float")
                dtype_ok = true; // normalized string form (level2 writes "torch.float" for dtype=float32)
        }
        if (!dtype_ok)
            return false;

        // ncnn scalar BinaryOp reads and writes the blob through a float*;
        // it can only run over f32 storage. an integral input would be
        // reinterpreted as float bit patterns (int32), and one-byte bool/uint8
        // storage would be read/written four bytes at a time past its extent.
        // an explicit float output over an integral input would need a real
        // cast first, which this lowering does not provide, so decline those
        // too unless the input already has compatible floating-point storage.
        const std::map<std::string, const Operator*>::const_iterator opit = matched_operators.find("op_0");
        if (opit == matched_operators.end())
            return false;
        const Operator* op0 = opit->second;
        if (op0->inputs.empty())
            return false;
        if (op0->inputs[0]->type != 1) // only f32 input storage is safe here
            return false;

        return true;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_ones_like, 19)

} // namespace ncnn

} // namespace pnnx
