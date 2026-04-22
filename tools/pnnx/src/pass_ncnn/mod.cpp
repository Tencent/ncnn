// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class onnx_Mod : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 A
pnnx.Input              input_1     0 1 B
Mod                     op_0        2 1 A B C fmod=%fmod
pnnx.Output             output      1 0 C
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Mod";
    }

    const char* name_str() const
    {
        return "mod";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int fmod = 0;
        if (captured_params.find("fmod") != captured_params.end())
        {
            const Parameter& fmod_p = captured_params.at("fmod");
            if (fmod_p.type == 1)
                fmod = fmod_p.b ? 1 : 0;
            else if (fmod_p.type == 2)
                fmod = fmod_p.i;
        }

        op->params["0"] = fmod;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(onnx_Mod, 20)

} // namespace ncnn

} // namespace pnnx
