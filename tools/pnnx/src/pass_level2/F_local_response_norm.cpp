// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "pass_level2.h"

namespace pnnx {

class F_local_response_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
15 14
pnnx.Input              input       0 1 input
aten::mul               op_0        2 1 input input 6
torch.unsqueeze         op_1        1 1 6 input.1 dim=1
F.pad                   op_2        1 1 input.1 div.1 mode=constant pad=(0,0,%pad_left,%pad_right) value=%padzero
F.avg_pool2d            op_3        1 1 div.1 25 kernel_size=(%size,1) stride=(1,1) padding=(0,0) ceil_mode=False count_include_pad=True divisor_override=None
torch.squeeze           op_4        1 1 25 div0.1 dim=1
prim::Constant          op_5        0 1 alpha value=%alpha
aten::mul               op_6        2 1 div0.1 alpha 30
prim::Constant          op_7        0 1 k value=%k
prim::Constant          op_8        0 1 62 value=1
aten::add               op_9        3 1 30 k 62 33
prim::Constant          op_10       0 1 beta value=%beta
aten::pow               op_11       2 1 33 beta div1.1
aten::div               op_12       2 1 input div1.1 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.local_response_norm";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("padzero").type == 0)
            return true;

        if (captured_params.at("padzero").type == 2)
            return captured_params.at("padzero").i == 0;

        if (captured_params.at("padzero").type == 3)
            return captured_params.at("padzero").f == 0.f;

        return false;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["size"] = captured_params.at("size");
        op->params["alpha"] = captured_params.at("alpha");
        op->params["beta"] = captured_params.at("beta");
        op->params["k"] = captured_params.at("k");
    }
};

class F_local_response_norm_1 : public F_local_response_norm
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
24 23
pnnx.Input              input       0 1 input
aten::mul               op_0        2 1 input input div7.1
Tensor.size             op_1        1 1 input 77 dim=0
Tensor.size             op_2        1 1 input 86 dim=1
Tensor.size             op_3        1 1 input 95 dim=2
Tensor.size             op_4        1 1 input 104 dim=3
prim::Constant          op_5        0 1 434 value=1
prim::Constant          op_6        0 1 113 value=-1
prim::ListConstruct     op_7        5 1 77 434 86 95 113 114
Tensor.view             op_8        2 1 div7.1 114 input2.1
F.pad                   op_9        1 1 input2.1 div8.1 mode=constant pad=(0,0,0,0,%pad_left,%pad_right) value=%padzero
F.avg_pool3d            op_10       1 1 div8.1 129 ceil_mode=False count_include_pad=True divisor_override=None kernel_size=(%size,1,1) padding=(0,0,0) stride=(1,1,1)
torch.squeeze           op_11       1 1 129 div9.1 dim=1
prim::ListConstruct     op_12       4 1 77 86 95 104 137
Tensor.view             op_13       2 1 div9.1 137 div10.1
prim::Constant          op_14       0 1 alpha value=%alpha
aten::mul               op_15       2 1 div10.1 alpha 141
prim::Constant          op_16       0 1 k value=%k
prim::Constant          op_17       0 1 457 value=1
aten::add               op_18       3 1 141 k 457 144
prim::Constant          op_19       0 1 beta value=%beta
aten::pow               op_20       2 1 144 beta div11.1
aten::div               op_21       2 1 input div11.1 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class F_local_response_norm_2 : public F_local_response_norm
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
25 24
pnnx.Input              input       0 1 input
aten::mul               op_0        2 1 input input div17.1
Tensor.size             op_1        1 1 input 230 dim=0
Tensor.size             op_2        1 1 input 239 dim=1
Tensor.size             op_3        1 1 input 248 dim=2
Tensor.size             op_4        1 1 input 257 dim=3
Tensor.size             op_5        1 1 input 263 dim=4
prim::Constant          op_6        0 1 492 value=1
prim::Constant          op_7        0 1 493 value=-1
prim::ListConstruct     op_8        5 1 230 492 239 248 493 272
Tensor.view             op_9        2 1 div17.1 272 input5.1
F.pad                   op_10       1 1 input5.1 div18.1 mode=constant pad=(0,0,0,0,%pad_left,%pad_right) value=%padzero
F.avg_pool3d            op_11       1 1 div18.1 286 ceil_mode=False count_include_pad=True divisor_override=None kernel_size=(%size,1,1) padding=(0,0,0) stride=(1,1,1)
torch.squeeze           op_12       1 1 286 div19.1 dim=1
prim::ListConstruct     op_13       5 1 230 239 248 257 263 295
Tensor.view             op_14       2 1 div19.1 295 div20.1
prim::Constant          op_15       0 1 alpha value=%alpha
aten::mul               op_16       2 1 div20.1 alpha 299
prim::Constant          op_17       0 1 k value=%k
prim::Constant          op_18       0 1 517 value=1
aten::add               op_19       3 1 299 k 517 302
prim::Constant          op_20       0 1 beta value=%beta
aten::pow               op_21       2 1 302 beta div21.1
aten::div               op_22       2 1 input div21.1 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_local_response_norm, 130)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_local_response_norm_1, 130)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_local_response_norm_2, 130)

} // namespace pnnx
