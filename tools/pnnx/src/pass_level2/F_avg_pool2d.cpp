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

class F_avg_pool2d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 kernel_size
pnnx.Input              input_2     0 1 stride
pnnx.Input              input_3     0 1 padding
prim::Constant          op_0        0 1 ceil_mode value=%ceil_mode
prim::Constant          op_1        0 1 count_include_pad value=%count_include_pad
prim::Constant          op_2        0 1 divisor_override value=%divisor_override
aten::avg_pool2d        op_3        7 1 input kernel_size stride padding ceil_mode count_include_pad divisor_override out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.avg_pool2d";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_avg_pool2d, 10)

// https://github.com/pytorch/pytorch/blob/c263bd43e8e8502d4726643bc6fd046f0130ac0e/torch/onnx/symbolic_opset9.py#L1496
static int get_pool_ceil_padding(int w, int ksize, int stride, int pad)
{
    if (stride == 1)
        return 0;

    int ceiled_output_w = int(ceil((w + pad * 2 - ksize) / float(stride))) + 1;

    if ((ceiled_output_w - 1) * stride >= w + pad)
        ceiled_output_w = ceiled_output_w - 1;

    int ceil_pad = ksize - (w + pad * 2 - ((ceiled_output_w - 1) * stride + 1));

    return ceil_pad;
}

class F_avg_pool2d_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
AveragePool             op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.avg_pool2d";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        if (captured_params.find("op_0.kernel_shape") == captured_params.end())
            return false;

        if (captured_params.at("op_0.kernel_shape").type != 5 || captured_params.at("op_0.kernel_shape").ai.size() != 2)
            return false;

        if (captured_params.find("op_0.dilations") != captured_params.end())
        {
            if (captured_params.at("op_0.dilations").type != 5 || captured_params.at("op_0.dilations").ai.size() != 2)
                return false;
        }

        if (captured_params.find("op_0.strides") != captured_params.end())
        {
            if (captured_params.at("op_0.strides").type != 5 || captured_params.at("op_0.strides").ai.size() != 2)
                return false;
        }

        if (captured_params.find("op_0.pads") != captured_params.end())
        {
            if (captured_params.at("op_0.pads").type != 5 || captured_params.at("op_0.pads").ai.size() != 4)
                return false;

            const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
            const int ceil_mode = captured_params.find("op_0.ceil_mode") != captured_params.end() ? captured_params.at("op_0.ceil_mode").i : 0;
            if (pads[0] != pads[2] || pads[1] != pads[3])
            {
                // ceil_mode for opset9
                const Operator* avgpool = matched_operators.at("op_0");
                const std::vector<int>& in_shape = avgpool->inputs[0]->shape;
                if (in_shape.size() < 2)
                    return false;

                const int inh = in_shape[in_shape.size() - 2];
                const int inw = in_shape[in_shape.size() - 1];
                const int kh = captured_params.at("op_0.kernel_shape").ai[0];
                const int kw = captured_params.at("op_0.kernel_shape").ai[1];
                const int dh = captured_params.find("op_0.dilations") != captured_params.end() ? captured_params.at("op_0.dilations").ai[0] : 1;
                const int dw = captured_params.find("op_0.dilations") != captured_params.end() ? captured_params.at("op_0.dilations").ai[1] : 1;
                const int sh = captured_params.find("op_0.strides") != captured_params.end() ? captured_params.at("op_0.strides").ai[0] : 1;
                const int sw = captured_params.find("op_0.strides") != captured_params.end() ? captured_params.at("op_0.strides").ai[1] : 1;

                const int keh = dh * (kh - 1) + 1;
                const int kew = dw * (kw - 1) + 1;

                int ceil_padh = get_pool_ceil_padding(inh, keh, sh, pads[0]);
                int ceil_padw = get_pool_ceil_padding(inw, kew, sw, pads[1]);

                if (ceil_mode == 0 && pads[0] + ceil_padh == pads[2] && pads[1] + ceil_padw == pads[3])
                {
                    // useless tail padding  :D
                }
                else
                {
                    return false;
                }
            }
        }

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["kernel_size"] = captured_params.at("op_0.kernel_shape");

        if (captured_params.find("op_0.dilations") != captured_params.end())
        {
            op->params["dilation"] = captured_params.at("op_0.dilations");
        }

        if (captured_params.find("op_0.strides") != captured_params.end())
        {
            op->params["stride"] = captured_params.at("op_0.strides");
        }
        else
        {
            op->params["stride"] = {1, 1};
        }

        if (captured_params.find("op_0.pads") != captured_params.end())
        {
            const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
            op->params["padding"] = {pads[0], pads[1]};
        }
        else
        {
            op->params["padding"] = {0, 0};
        }

        if (captured_params.find("op_0.count_include_pad") != captured_params.end())
        {
            int count_include_pad = captured_params.at("op_0.count_include_pad").i;
            op->params["count_include_pad"] = (count_include_pad != 0);
        }
        else
        {
            op->params["count_include_pad"] = false;
        }

        if (captured_params.find("op_0.ceil_mode") != captured_params.end())
        {
            int ceil_mode = captured_params.at("op_0.ceil_mode").i;
            op->params["ceil_mode"] = (ceil_mode != 0);
        }
        else
        {
            op->params["ceil_mode"] = false;
        }

        // ceil_mode for opset9
        if (captured_params.find("op_0.pads") != captured_params.end())
        {
            const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
            if (pads[0] != pads[2] || pads[1] != pads[3])
            {
                op->params["ceil_mode"] = true;
            }
        }

        op->params["divisor_override"] = Parameter();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_avg_pool2d_onnx, 10)

} // namespace pnnx
