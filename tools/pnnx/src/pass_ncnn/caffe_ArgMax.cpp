// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class caffe_ArgMax : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
caffe.ArgMax            op_0        1 1 input out top_k=%top_k mode=0 shape=%shape
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
Flatten                 flatten     1 1 input flat
ArgMax                  argmax      1 1 flat indices 0=0 1=%top_k
Reshape                 reshape     1 1 indices out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);
        std::vector<int> shape = captured_params.at("shape").ai;
        if (shape.size() > 1)
            shape.erase(shape.begin());
        Operator* reshape = ops.at("reshape");
        reshape->params["0"] = shape.back();
        if (shape.size() >= 2)
            reshape->params["1"] = shape[shape.size() - 2];
        if (shape.size() >= 3)
            reshape->params["2"] = shape[0];
        if (shape.size() == 4)
            reshape->params["11"] = shape[1];
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(caffe_ArgMax, 20)

class caffe_ArgMax_values : public caffe_ArgMax
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
caffe.ArgMax            op_0        1 1 input out top_k=%top_k mode=1 shape=%shape
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
Flatten                 flatten     1 1 input flat
ArgMax                  argmax      1 1 flat pairs 0=1 1=%top_k
Crop                    values      1 1 pairs values 9=(0) 10=(1) 11=(0)
Reshape                 reshape     1 1 values out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(caffe_ArgMax_values, 20)

class caffe_ArgMax_pairs : public caffe_ArgMax
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
caffe.ArgMax            op_0        1 1 input out top_k=%top_k mode=2 shape=%shape
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 7
pnnx.Input              input       0 1 input
Flatten                 flatten     1 1 input flat
ArgMax                  argmax      1 1 flat pairs 0=1 1=%top_k
Slice                   split       1 2 pairs values indices 0=(1,1) 1=0
Concat                  reorder     2 1 indices values ordered 0=0
Reshape                 reshape     1 1 ordered out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(caffe_ArgMax_pairs, 20)

} // namespace ncnn

} // namespace pnnx
