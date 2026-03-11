// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_deconvolutiondepthwise3d_activation.h"

#include "pass_level2.h"

#include <float.h>

namespace pnnx {

namespace ncnn {

class fuse_deconvolutiondepthwise3d_relu_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
DeconvolutionDepthWise3D op_0       1 1 input a %*=%*
ReLU                    op_1        1 1 a out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "DeconvolutionDepthWise3D";
    }

    const char* name_str() const
    {
        return "deconvdw3drelu";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.find("op_0.9") == captured_params.end();
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        for (const auto& p : captured_params)
        {
            const std::string& pkey = p.first;
            const Parameter& pp = p.second;

            if (pkey.substr(0, 5) == "op_0.")
                op->params[pkey.substr(5)] = pp;
        }

        for (const auto& a : captured_attrs)
        {
            const std::string& akey = a.first;
            const Attribute& ap = a.second;

            if (akey.substr(0, 5) == "op_0.")
                op->attrs[akey.substr(5)] = ap;
        }

        float slope = 0.f;
        if (captured_params.find("op_1.0") != captured_params.end())
        {
            slope = captured_params.at("op_1.0").f;
        }

        if (slope == 0.f)
        {
            op->params["9"] = 1;
        }
        else
        {
            op->params["9"] = 2;
            op->params["10"] = Parameter{slope};
        }
    }
};

class fuse_deconvolutiondepthwise3d_clip_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
DeconvolutionDepthWise3D op_0       1 1 input a %*=%*
Clip                    op_1        1 1 a out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "DeconvolutionDepthWise3D";
    }

    const char* name_str() const
    {
        return "deconvdw3dclip";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.find("op_0.9") == captured_params.end();
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        for (const auto& p : captured_params)
        {
            const std::string& pkey = p.first;
            const Parameter& pp = p.second;

            if (pkey.substr(0, 5) == "op_0.")
                op->params[pkey.substr(5)] = pp;
        }

        for (const auto& a : captured_attrs)
        {
            const std::string& akey = a.first;
            const Attribute& ap = a.second;

            if (akey.substr(0, 5) == "op_0.")
                op->attrs[akey.substr(5)] = ap;
        }

        float min = -FLT_MAX;
        float max = FLT_MAX;
        if (captured_params.find("op_1.0") != captured_params.end())
        {
            min = captured_params.at("op_1.0").f;
        }
        if (captured_params.find("op_1.1") != captured_params.end())
        {
            max = captured_params.at("op_1.1").f;
        }

        op->params["9"] = 3;
        op->params["10"] = Parameter{min, max};
    }
};

class fuse_deconvolutiondepthwise3d_sigmoid_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
DeconvolutionDepthWise3D op_0       1 1 input a %*=%*
Sigmoid                 op_1        1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "DeconvolutionDepthWise3D";
    }

    const char* name_str() const
    {
        return "deconvdw3dsigmoid";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.find("op_0.9") == captured_params.end();
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        for (const auto& p : captured_params)
        {
            const std::string& pkey = p.first;
            const Parameter& pp = p.second;

            if (pkey.substr(0, 5) == "op_0.")
                op->params[pkey.substr(5)] = pp;
        }

        for (const auto& a : captured_attrs)
        {
            const std::string& akey = a.first;
            const Attribute& ap = a.second;

            if (akey.substr(0, 5) == "op_0.")
                op->attrs[akey.substr(5)] = ap;
        }

        op->params["9"] = 4;
    }
};

class fuse_deconvolutiondepthwise3d_mish_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
DeconvolutionDepthWise3D op_0       1 1 input a %*=%*
Mish                    op_1        1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "DeconvolutionDepthWise3D";
    }

    const char* name_str() const
    {
        return "deconvdw3dmish";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.find("op_0.9") == captured_params.end();
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        for (const auto& p : captured_params)
        {
            const std::string& pkey = p.first;
            const Parameter& pp = p.second;

            if (pkey.substr(0, 5) == "op_0.")
                op->params[pkey.substr(5)] = pp;
        }

        for (const auto& a : captured_attrs)
        {
            const std::string& akey = a.first;
            const Attribute& ap = a.second;

            if (akey.substr(0, 5) == "op_0.")
                op->attrs[akey.substr(5)] = ap;
        }

        op->params["9"] = 5;
    }
};

class fuse_deconvolutiondepthwise3d_hardswish_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
DeconvolutionDepthWise3D op_0       1 1 input a %*=%*
HardSwish               op_1        1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "DeconvolutionDepthWise3D";
    }

    const char* name_str() const
    {
        return "deconvdw3dhardswish";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.find("op_0.9") == captured_params.end();
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        for (const auto& p : captured_params)
        {
            const std::string& pkey = p.first;
            const Parameter& pp = p.second;

            if (pkey.substr(0, 5) == "op_0.")
                op->params[pkey.substr(5)] = pp;
        }

        for (const auto& a : captured_attrs)
        {
            const std::string& akey = a.first;
            const Attribute& ap = a.second;

            if (akey.substr(0, 5) == "op_0.")
                op->attrs[akey.substr(5)] = ap;
        }

        op->params["9"] = 6;
        op->params["10"] = Parameter{1.f / 6, 0.5f};
    }
};

class fuse_deconvolutiondepthwise3d_gelu_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
DeconvolutionDepthWise3D op_0       1 1 input a %*=%*
GELU                    op_1        1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "DeconvolutionDepthWise3D";
    }

    const char* name_str() const
    {
        return "deconvdw3dgelu";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        // match GELU without parameters (fast_gelu = 0)
        if (captured_params.find("op_0.9") != captured_params.end())
            return false;
        if (captured_params.find("op_1.0") != captured_params.end())
            return false;
        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        for (const auto& p : captured_params)
        {
            const std::string& pkey = p.first;
            const Parameter& pp = p.second;

            if (pkey.substr(0, 5) == "op_0.")
                op->params[pkey.substr(5)] = pp;
        }

        for (const auto& a : captured_attrs)
        {
            const std::string& akey = a.first;
            const Attribute& ap = a.second;

            if (akey.substr(0, 5) == "op_0.")
                op->attrs[akey.substr(5)] = ap;
        }

        op->params["9"] = 7;
        op->params["10"] = Parameter{0};
    }
};

class fuse_deconvolutiondepthwise3d_gelu_fast_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
DeconvolutionDepthWise3D op_0       1 1 input a %*=%*
GELU                    op_1        1 1 a out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "DeconvolutionDepthWise3D";
    }

    const char* name_str() const
    {
        return "deconvdw3dgelu";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        // match GELU with parameter 0=1 (fast_gelu = 1)
        if (captured_params.find("op_0.9") != captured_params.end())
            return false;
        if (captured_params.find("op_1.0") != captured_params.end())
        {
            const Parameter& fast_gelu = captured_params.at("op_1.0");
            if (fast_gelu.type == 2 && fast_gelu.i == 1)
                return true;
        }
        return false;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        for (const auto& p : captured_params)
        {
            const std::string& pkey = p.first;
            const Parameter& pp = p.second;

            if (pkey.substr(0, 5) == "op_0.")
                op->params[pkey.substr(5)] = pp;
        }

        for (const auto& a : captured_attrs)
        {
            const std::string& akey = a.first;
            const Attribute& ap = a.second;

            if (akey.substr(0, 5) == "op_0.")
                op->attrs[akey.substr(5)] = ap;
        }

        op->params["9"] = 7;
        op->params["10"] = Parameter{1}; // fast_gelu = 1
    }
};

class fuse_deconvolutiondepthwise3d_silu_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
DeconvolutionDepthWise3D op_0       1 1 input a %*=%*
nn.SiLU                 op_1        1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "DeconvolutionDepthWise3D";
    }

    const char* name_str() const
    {
        return "deconvdw3dsilu";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.find("op_0.9") == captured_params.end();
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        for (const auto& p : captured_params)
        {
            const std::string& pkey = p.first;
            const Parameter& pp = p.second;

            if (pkey.substr(0, 5) == "op_0.")
                op->params[pkey.substr(5)] = pp;
        }

        for (const auto& a : captured_attrs)
        {
            const std::string& akey = a.first;
            const Attribute& ap = a.second;

            if (akey.substr(0, 5) == "op_0.")
                op->attrs[akey.substr(5)] = ap;
        }

        op->params["9"] = 8;
    }
};

class fuse_deconvolutiondepthwise3d_elu_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
DeconvolutionDepthWise3D op_0       1 1 input a %*=%*
ELU                     op_1        1 1 a out alpha=%alpha
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "DeconvolutionDepthWise3D";
    }

    const char* name_str() const
    {
        return "deconvdw3delu";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.find("op_0.9") == captured_params.end();
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        for (const auto& p : captured_params)
        {
            const std::string& pkey = p.first;
            const Parameter& pp = p.second;

            if (pkey.substr(0, 5) == "op_0.")
                op->params[pkey.substr(5)] = pp;
        }

        for (const auto& a : captured_attrs)
        {
            const std::string& akey = a.first;
            const Attribute& ap = a.second;

            if (akey.substr(0, 5) == "op_0.")
                op->attrs[akey.substr(5)] = ap;
        }

        op->params["9"] = 9;
        op->params["10"] = captured_params.at("alpha");
    }
};

class fuse_deconvolutiondepthwise3d_selu_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
DeconvolutionDepthWise3D op_0       1 1 input a %*=%*
nn.SELU                 op_1        1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "DeconvolutionDepthWise3D";
    }

    const char* name_str() const
    {
        return "deconvdw3dselu";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.find("op_0.9") == captured_params.end();
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        for (const auto& p : captured_params)
        {
            const std::string& pkey = p.first;
            const Parameter& pp = p.second;

            if (pkey.substr(0, 5) == "op_0.")
                op->params[pkey.substr(5)] = pp;
        }

        for (const auto& a : captured_attrs)
        {
            const std::string& akey = a.first;
            const Attribute& ap = a.second;

            if (akey.substr(0, 5) == "op_0.")
                op->attrs[akey.substr(5)] = ap;
        }

        op->params["9"] = 10;
    }
};

void fuse_deconvolutiondepthwise3d_activation(Graph& graph)
{
    fuse_deconvolutiondepthwise3d_relu_pass a;
    fuse_deconvolutiondepthwise3d_clip_pass b;
    fuse_deconvolutiondepthwise3d_sigmoid_pass c;
    fuse_deconvolutiondepthwise3d_mish_pass d;
    fuse_deconvolutiondepthwise3d_hardswish_pass e;
    fuse_deconvolutiondepthwise3d_gelu_pass f;
    fuse_deconvolutiondepthwise3d_gelu_fast_pass f2;
    fuse_deconvolutiondepthwise3d_silu_pass g;
    fuse_deconvolutiondepthwise3d_elu_pass h;
    fuse_deconvolutiondepthwise3d_selu_pass i;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
    pnnx_graph_rewrite(graph, &c, opindex);
    pnnx_graph_rewrite(graph, &d, opindex);
    pnnx_graph_rewrite(graph, &e, opindex);
    pnnx_graph_rewrite(graph, &f, opindex);
    pnnx_graph_rewrite(graph, &f2, opindex);
    pnnx_graph_rewrite(graph, &g, opindex);
    pnnx_graph_rewrite(graph, &h, opindex);
    pnnx_graph_rewrite(graph, &i, opindex);
}

} // namespace ncnn

} // namespace pnnx
