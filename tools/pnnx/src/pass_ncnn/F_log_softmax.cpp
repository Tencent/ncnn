// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "log_softmax.h"

namespace pnnx {

namespace ncnn {

class F_log_softmax : public LogSoftmaxPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.log_softmax           op          1 1 input out dim=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F_log_softmax";
    }

    const char* name_str() const
    {
        return "f_logsoftmax";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_log_softmax, 19)

} // namespace ncnn

} // namespace pnnx
