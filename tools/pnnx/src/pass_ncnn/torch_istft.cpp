// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_istft : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
torch.view_as_complex   op_0        1 1 input a
torch.istft             op_1        1 1 a out center=%center hop_length=%hop_length length=%length n_fft=%n_fft normalized=%normalized onesided=%onesided return_complex=False win_length=%win_length window=None
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "InverseSpectrogram";
    }

    const char* name_str() const
    {
        return "istft";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        // op->params["0"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_istft, 20)

class torch_istft_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
torch.view_as_complex   op_0        1 1 input a
torch.istft             op_1        1 1 a b center=%center hop_length=%hop_length length=%length n_fft=%n_fft normalized=%normalized onesided=%onesided return_complex=True win_length=%win_length window=None
torch.view_as_real      op_2        1 1 b out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "InverseSpectrogram";
    }

    const char* name_str() const
    {
        return "istft";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        // op->params["0"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_istft_1, 20)

} // namespace ncnn

} // namespace pnnx
