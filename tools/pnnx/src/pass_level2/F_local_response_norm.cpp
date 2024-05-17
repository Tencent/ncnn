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
35 34
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 k value=%k
prim::Constant          op_1        0 1 alpha value=%alpha
prim::Constant          op_2        0 1 24 value=None
prim::Constant          op_3        0 1 23 value=True
prim::Constant          op_4        0 1 22 value=False
prim::Constant          op_5        0 1 7 value=1
prim::Constant          op_6        0 1 10 value=0
prim::Constant          op_7        0 1 size value=%size
prim::Constant          op_8        0 1 beta value=%beta
aten::mul               op_9        2 1 input input 6
aten::unsqueeze         op_10       2 1 6 7 input.1
prim::Constant          op_11       0 1 52 value=0
prim::Constant          op_12       0 1 53 value=*
prim::Constant          op_13       0 1 54 value=*
prim::ListConstruct     op_14       4 1 10 52 53 54 11
prim::Constant          op_15       0 1 55 value=%padzero
aten::constant_pad_nd   op_16       3 1 input.1 11 55 div.1
prim::Constant          op_17       0 1 56 value=1
prim::ListConstruct     op_18       2 1 size 56 16
prim::Constant          op_19       0 1 57 value=1
prim::Constant          op_20       0 1 58 value=1
prim::ListConstruct     op_21       2 1 57 58 17
prim::Constant          op_22       0 1 59 value=0
prim::Constant          op_23       0 1 60 value=0
prim::ListConstruct     op_24       2 1 59 60 18
aten::avg_pool2d        op_25       7 1 div.1 16 17 18 22 23 24 25
prim::Constant          op_26       0 1 61 value=1
aten::squeeze           op_27       2 1 25 61 div0.1
aten::mul               op_28       2 1 div0.1 alpha 30
prim::Constant          op_29       0 1 62 value=1
aten::add               op_30       3 1 30 k 62 33
aten::pow               op_31       2 1 33 beta div1.1
aten::div               op_32       2 1 input div1.1 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.local_response_norm";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_local_response_norm, 8)

class F_local_response_norm_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
65 64
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 k value=%k
prim::Constant          op_1        0 1 alpha value=%alpha
prim::Constant          op_2        0 1 66 value=None
prim::Constant          op_3        0 1 65 value=True
prim::Constant          op_4        0 1 64 value=False
prim::Constant          op_5        0 1 7 value=1
prim::Constant          op_6        0 1 10 value=0
prim::Constant          op_7        0 1 29 value=2
prim::Constant          op_8        0 1 39 value=3
prim::Constant          op_9        0 1 49 value=-1
prim::Constant          op_10       0 1 size value=%size
prim::Constant          op_11       0 1 beta value=%beta
aten::mul               op_12       2 1 input input 6
aten::unsqueeze         op_13       2 1 6 7 div.1
aten::size              op_14       2 1 input 10 11
prim::NumToTensor       op_15       1 1 11 12
aten::Int               op_16       1 1 12 15
aten::Int               op_17       1 1 12 18
prim::Constant          op_18       0 1 101 value=1
aten::size              op_19       2 1 input 101 20
prim::NumToTensor       op_20       1 1 20 21
aten::Int               op_21       1 1 21 24
aten::Int               op_22       1 1 21 27
aten::size              op_23       2 1 input 29 30
prim::NumToTensor       op_24       1 1 30 31
aten::Int               op_25       1 1 31 34
aten::Int               op_26       1 1 31 37
aten::size              op_27       2 1 input 39 40
prim::NumToTensor       op_28       1 1 40 41
aten::Int               op_29       1 1 41 44
prim::Constant          op_30       0 1 102 value=1
prim::ListConstruct     op_31       5 1 18 102 27 37 49 50
aten::view              op_32       2 1 div.1 50 input.1
prim::Constant          op_33       0 1 103 value=0
prim::Constant          op_34       0 1 104 value=0
prim::Constant          op_35       0 1 105 value=0
prim::Constant          op_36       0 1 106 value=0
prim::Constant          op_37       0 1 107 value=*
prim::Constant          op_38       0 1 108 value=*
prim::ListConstruct     op_39       6 1 103 104 105 106 107 108 53
prim::Constant          op_40       0 1 109 value=%padzero
aten::constant_pad_nd   op_41       3 1 input.1 53 109 div0.1
prim::Constant          op_42       0 1 110 value=1
prim::Constant          op_43       0 1 111 value=1
prim::ListConstruct     op_44       3 1 size 110 111 58
prim::Constant          op_45       0 1 112 value=1
prim::Constant          op_46       0 1 113 value=1
prim::Constant          op_47       0 1 114 value=1
prim::ListConstruct     op_48       3 1 112 113 114 59
prim::Constant          op_49       0 1 115 value=0
prim::Constant          op_50       0 1 116 value=0
prim::Constant          op_51       0 1 117 value=0
prim::ListConstruct     op_52       3 1 115 116 117 60
aten::avg_pool3d        op_53       7 1 div0.1 58 59 60 64 65 66 67
prim::Constant          op_54       0 1 118 value=1
aten::squeeze           op_55       2 1 67 118 div1.1
prim::ListConstruct     op_56       4 1 15 24 34 44 75
aten::view              op_57       2 1 div1.1 75 div2.1
aten::mul               op_58       2 1 div2.1 alpha 79
prim::Constant          op_59       0 1 119 value=1
aten::add               op_60       3 1 79 k 119 82
aten::pow               op_61       2 1 82 beta div3.1
aten::div               op_62       2 1 input div3.1 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.local_response_norm";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_local_response_norm_1, 8)

class F_local_response_norm_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
69 68
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 k value=%k
prim::Constant          op_1        0 1 alpha value=%alpha
prim::Constant          op_2        0 1 73 value=None
prim::Constant          op_3        0 1 72 value=True
prim::Constant          op_4        0 1 71 value=False
prim::Constant          op_5        0 1 7 value=1
prim::Constant          op_6        0 1 10 value=0
prim::Constant          op_7        0 1 29 value=2
prim::Constant          op_8        0 1 39 value=3
prim::Constant          op_9        0 1 46 value=4
prim::Constant          op_10       0 1 56 value=-1
prim::Constant          op_11       0 1 size value=%size
prim::Constant          op_12       0 1 beta value=%beta
aten::mul               op_13       2 1 input input 6
aten::unsqueeze         op_14       2 1 6 7 div.1
aten::size              op_15       2 1 input 10 11
prim::NumToTensor       op_16       1 1 11 12
aten::Int               op_17       1 1 12 15
aten::Int               op_18       1 1 12 18
prim::Constant          op_19       0 1 109 value=1
aten::size              op_20       2 1 input 109 20
prim::NumToTensor       op_21       1 1 20 21
aten::Int               op_22       1 1 21 24
aten::Int               op_23       1 1 21 27
aten::size              op_24       2 1 input 29 30
prim::NumToTensor       op_25       1 1 30 31
aten::Int               op_26       1 1 31 34
aten::Int               op_27       1 1 31 37
aten::size              op_28       2 1 input 39 40
prim::NumToTensor       op_29       1 1 40 41
aten::Int               op_30       1 1 41 44
aten::size              op_31       2 1 input 46 47
prim::NumToTensor       op_32       1 1 47 48
aten::Int               op_33       1 1 48 51
prim::Constant          op_34       0 1 110 value=1
prim::ListConstruct     op_35       5 1 18 110 27 37 56 57
aten::view              op_36       2 1 div.1 57 input.1
prim::Constant          op_37       0 1 111 value=0
prim::Constant          op_38       0 1 112 value=0
prim::Constant          op_39       0 1 113 value=0
prim::Constant          op_40       0 1 114 value=0
prim::Constant          op_41       0 1 115 value=*
prim::Constant          op_42       0 1 116 value=*
prim::ListConstruct     op_43       6 1 111 112 113 114 115 116 60
prim::Constant          op_44       0 1 117 value=%padzero
aten::constant_pad_nd   op_45       3 1 input.1 60 117 div0.1
prim::Constant          op_46       0 1 118 value=1
prim::Constant          op_47       0 1 119 value=1
prim::ListConstruct     op_48       3 1 size 118 119 65
prim::Constant          op_49       0 1 120 value=1
prim::Constant          op_50       0 1 121 value=1
prim::Constant          op_51       0 1 122 value=1
prim::ListConstruct     op_52       3 1 120 121 122 66
prim::Constant          op_53       0 1 123 value=0
prim::Constant          op_54       0 1 124 value=0
prim::Constant          op_55       0 1 125 value=0
prim::ListConstruct     op_56       3 1 123 124 125 67
aten::avg_pool3d        op_57       7 1 div0.1 65 66 67 71 72 73 74
prim::Constant          op_58       0 1 126 value=1
aten::squeeze           op_59       2 1 74 126 div1.1
prim::ListConstruct     op_60       5 1 15 24 34 44 51 83
aten::view              op_61       2 1 div1.1 83 div2.1
aten::mul               op_62       2 1 div2.1 alpha 87
prim::Constant          op_63       0 1 127 value=1
aten::add               op_64       3 1 87 k 127 90
aten::pow               op_65       2 1 90 beta div3.1
aten::div               op_66       2 1 input div3.1 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.local_response_norm";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_local_response_norm_2, 8)

class F_local_response_norm_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
36 35
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 k value=%k
prim::Constant          op_1        0 1 alpha value=%alpha
prim::Constant          op_2        0 1 24 value=None
prim::Constant          op_3        0 1 23 value=True
prim::Constant          op_4        0 1 22 value=False
prim::Constant          op_5        0 1 7 value=1
prim::Constant          op_6        0 1 10 value=0
prim::Constant          op_7        0 1 size value=%size
prim::Constant          op_8        0 1 beta value=%beta
aten::mul               op_9        2 1 input input 6
aten::unsqueeze         op_10       2 1 6 7 input.1
prim::Constant          op_11       0 1 52 value=0
prim::Constant          op_12       0 1 53 value=*
prim::Constant          op_13       0 1 54 value=*
prim::ListConstruct     op_14       4 1 10 52 53 54 11
prim::Constant          op_15       0 1 12 value=constant
prim::Constant          op_16       0 1 55 value=%padzero
aten::pad               op_17       4 1 input.1 11 12 55 div.1
prim::Constant          op_18       0 1 56 value=1
prim::ListConstruct     op_19       2 1 size 56 16
prim::Constant          op_20       0 1 57 value=1
prim::Constant          op_21       0 1 58 value=1
prim::ListConstruct     op_22       2 1 57 58 17
prim::Constant          op_23       0 1 59 value=0
prim::Constant          op_24       0 1 60 value=0
prim::ListConstruct     op_25       2 1 59 60 18
aten::avg_pool2d        op_26       7 1 div.1 16 17 18 22 23 24 25
prim::Constant          op_27       0 1 61 value=1
aten::squeeze           op_28       2 1 25 61 div0.1
aten::mul               op_29       2 1 div0.1 alpha 30
prim::Constant          op_30       0 1 62 value=1
aten::add               op_31       3 1 30 k 62 33
aten::pow               op_32       2 1 33 beta div1.1
aten::div               op_33       2 1 input div1.1 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.local_response_norm";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_local_response_norm_3, 8)

class F_local_response_norm_4 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
66 65
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 k value=%k
prim::Constant          op_1        0 1 alpha value=%alpha
prim::Constant          op_2        0 1 66 value=None
prim::Constant          op_3        0 1 65 value=True
prim::Constant          op_4        0 1 64 value=False
prim::Constant          op_5        0 1 7 value=1
prim::Constant          op_6        0 1 10 value=0
prim::Constant          op_7        0 1 29 value=2
prim::Constant          op_8        0 1 39 value=3
prim::Constant          op_9        0 1 49 value=-1
prim::Constant          op_10       0 1 size value=%size
prim::Constant          op_11       0 1 beta value=%beta
aten::mul               op_12       2 1 input input 6
aten::unsqueeze         op_13       2 1 6 7 div.1
aten::size              op_14       2 1 input 10 11
prim::NumToTensor       op_15       1 1 11 12
aten::Int               op_16       1 1 12 15
aten::Int               op_17       1 1 12 18
prim::Constant          op_18       0 1 101 value=1
aten::size              op_19       2 1 input 101 20
prim::NumToTensor       op_20       1 1 20 21
aten::Int               op_21       1 1 21 24
aten::Int               op_22       1 1 21 27
aten::size              op_23       2 1 input 29 30
prim::NumToTensor       op_24       1 1 30 31
aten::Int               op_25       1 1 31 34
aten::Int               op_26       1 1 31 37
aten::size              op_27       2 1 input 39 40
prim::NumToTensor       op_28       1 1 40 41
aten::Int               op_29       1 1 41 44
prim::Constant          op_30       0 1 102 value=1
prim::ListConstruct     op_31       5 1 18 102 27 37 49 50
aten::view              op_32       2 1 div.1 50 input.1
prim::Constant          op_33       0 1 103 value=0
prim::Constant          op_34       0 1 104 value=0
prim::Constant          op_35       0 1 105 value=0
prim::Constant          op_36       0 1 106 value=0
prim::Constant          op_37       0 1 107 value=*
prim::Constant          op_38       0 1 108 value=*
prim::ListConstruct     op_39       6 1 103 104 105 106 107 108 53
prim::Constant          op_40       0 1 54 value=constant
prim::Constant          op_41       0 1 109 value=%padzero
aten::pad               op_42       4 1 input.1 53 54 109 div0.1
prim::Constant          op_43       0 1 110 value=1
prim::Constant          op_44       0 1 111 value=1
prim::ListConstruct     op_45       3 1 size 110 111 58
prim::Constant          op_46       0 1 112 value=1
prim::Constant          op_47       0 1 113 value=1
prim::Constant          op_48       0 1 114 value=1
prim::ListConstruct     op_49       3 1 112 113 114 59
prim::Constant          op_50       0 1 115 value=0
prim::Constant          op_51       0 1 116 value=0
prim::Constant          op_52       0 1 117 value=0
prim::ListConstruct     op_53       3 1 115 116 117 60
aten::avg_pool3d        op_54       7 1 div0.1 58 59 60 64 65 66 67
prim::Constant          op_55       0 1 118 value=1
aten::squeeze           op_56       2 1 67 118 div1.1
prim::ListConstruct     op_57       4 1 15 24 34 44 75
aten::view              op_58       2 1 div1.1 75 div2.1
aten::mul               op_59       2 1 div2.1 alpha 79
prim::Constant          op_60       0 1 119 value=1
aten::add               op_61       3 1 79 k 119 82
aten::pow               op_62       2 1 82 beta div3.1
aten::div               op_63       2 1 input div3.1 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.local_response_norm";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_local_response_norm_4, 8)

class F_local_response_norm_5 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
70 69
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 k value=%k
prim::Constant          op_1        0 1 alpha value=%alpha
prim::Constant          op_2        0 1 73 value=None
prim::Constant          op_3        0 1 72 value=True
prim::Constant          op_4        0 1 71 value=False
prim::Constant          op_5        0 1 7 value=1
prim::Constant          op_6        0 1 10 value=0
prim::Constant          op_7        0 1 29 value=2
prim::Constant          op_8        0 1 39 value=3
prim::Constant          op_9        0 1 46 value=4
prim::Constant          op_10       0 1 56 value=-1
prim::Constant          op_11       0 1 size value=%size
prim::Constant          op_12       0 1 beta value=%beta
aten::mul               op_13       2 1 input input 6
aten::unsqueeze         op_14       2 1 6 7 div.1
aten::size              op_15       2 1 input 10 11
prim::NumToTensor       op_16       1 1 11 12
aten::Int               op_17       1 1 12 15
aten::Int               op_18       1 1 12 18
prim::Constant          op_19       0 1 109 value=1
aten::size              op_20       2 1 input 109 20
prim::NumToTensor       op_21       1 1 20 21
aten::Int               op_22       1 1 21 24
aten::Int               op_23       1 1 21 27
aten::size              op_24       2 1 input 29 30
prim::NumToTensor       op_25       1 1 30 31
aten::Int               op_26       1 1 31 34
aten::Int               op_27       1 1 31 37
aten::size              op_28       2 1 input 39 40
prim::NumToTensor       op_29       1 1 40 41
aten::Int               op_30       1 1 41 44
aten::size              op_31       2 1 input 46 47
prim::NumToTensor       op_32       1 1 47 48
aten::Int               op_33       1 1 48 51
prim::Constant          op_34       0 1 110 value=1
prim::ListConstruct     op_35       5 1 18 110 27 37 56 57
aten::view              op_36       2 1 div.1 57 input.1
prim::Constant          op_37       0 1 111 value=0
prim::Constant          op_38       0 1 112 value=0
prim::Constant          op_39       0 1 113 value=0
prim::Constant          op_40       0 1 114 value=0
prim::Constant          op_41       0 1 115 value=*
prim::Constant          op_42       0 1 116 value=*
prim::ListConstruct     op_43       6 1 111 112 113 114 115 116 60
prim::Constant          op_44       0 1 61 value=constant
prim::Constant          op_45       0 1 117 value=%padzero
aten::pad               op_46       4 1 input.1 60 61 117 div0.1
prim::Constant          op_47       0 1 118 value=1
prim::Constant          op_48       0 1 119 value=1
prim::ListConstruct     op_49       3 1 size 118 119 65
prim::Constant          op_50       0 1 120 value=1
prim::Constant          op_51       0 1 121 value=1
prim::Constant          op_52       0 1 122 value=1
prim::ListConstruct     op_53       3 1 120 121 122 66
prim::Constant          op_54       0 1 123 value=0
prim::Constant          op_55       0 1 124 value=0
prim::Constant          op_56       0 1 125 value=0
prim::ListConstruct     op_57       3 1 123 124 125 67
aten::avg_pool3d        op_58       7 1 div0.1 65 66 67 71 72 73 74
prim::Constant          op_59       0 1 126 value=1
aten::squeeze           op_60       2 1 74 126 div1.1
prim::ListConstruct     op_61       5 1 15 24 34 44 51 83
aten::view              op_62       2 1 div1.1 83 div2.1
aten::mul               op_63       2 1 div2.1 alpha 87
prim::Constant          op_64       0 1 127 value=1
aten::add               op_65       3 1 87 k 127 90
aten::pow               op_66       2 1 90 beta div3.1
aten::div               op_67       2 1 input div3.1 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.local_response_norm";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_local_response_norm_5, 8)

} // namespace pnnx
