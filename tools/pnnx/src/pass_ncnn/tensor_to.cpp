// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class Tensor_to : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 2
pnnx.Input              input_0     0 1 input
Tensor.to               op_0        1 1 input out copy=%copy dtype=%dtype
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Cast";
    }

    const char* name_str() const
    {
        return "to";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        // Map pnnx operand type (0=null 1=f32 2=f64 3=f16 4=i32 5=i64 7=i8 13=bf16)
        // to ncnn cast type (1=float32 2=float16 3=int8 4=bfloat16 5=int64 6=int32)
        static const int pnnx_to_ncnn_cast_type[] = {
            0, // 0=null
            1, // 1=f32  → ncnn float32
            1, // 2=f64  → ncnn float32 (no f64 in ncnn)
            2, // 3=f16  → ncnn float16
            6, // 4=i32  → ncnn int32
            5, // 5=i64  → ncnn int64
            0, // 6=i16  → unsupported
            3, // 7=i8   → ncnn int8
            0, // 8=u8   → unsupported
            0, // 9=bool → unsupported
            0, // 10=c64
            0, // 11=c128
            0, // 12=c32
            4, // 13=bf16 → ncnn bfloat16
        };

        const int in_pnnx_type = op->inputs[0]->type;
        int type_from = 0;
        if (in_pnnx_type >= 0 && in_pnnx_type <= 13)
            type_from = pnnx_to_ncnn_cast_type[in_pnnx_type];

        std::string dtype = "torch.float";
        if (captured_params.find("dtype") != captured_params.end())
        {
            dtype = captured_params.at("dtype").s;
        }

        int type_to = 0;
        if (dtype == "torch.float" || dtype == "torch.float32")
            type_to = 1;
        else if (dtype == "torch.float16" || dtype == "torch.half")
            type_to = 2;
        else if (dtype == "torch.int8")
            type_to = 3;
        else if (dtype == "torch.bfloat16")
            type_to = 4;
        else if (dtype == "torch.int64" || dtype == "torch.long")
            type_to = 5;
        else if (dtype == "torch.int32" || dtype == "torch.int")
            type_to = 6;

        op->params["0"] = type_from;
        op->params["1"] = type_to;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_to, 20)

} // namespace ncnn

} // namespace pnnx
