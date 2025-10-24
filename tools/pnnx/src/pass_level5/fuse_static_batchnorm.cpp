// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_static_batchnorm.h"

#include "pass_level2.h"

#include <math.h>
#include <string.h>

namespace pnnx {

class fuse_static_Fbatchnorm_pass_1d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_mean     0 1 running_mean @data=(%num_features)f32
pnnx.Attribute          op_var      0 1 running_var @data=(%num_features)f32
F.batchnorm             op_0        3 1 input running_mean running_var out weight=None bias=None eps=%eps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.BatchNorm1d          batchnorm   1 1 input out num_features=%num_features eps=%eps affine=False @running_mean=%op_mean.data @running_var=%op_var.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        size_t input_rank = matched_operators.at("op_0")->inputs[0]->shape.size();
        return input_rank == 2 || input_rank == 3;
    }
};

class fuse_static_Fbatchnorm_pass_1d_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
pnnx.Attribute          op_mean     0 1 running_mean @data=(%num_features)f32
pnnx.Attribute          op_var      0 1 running_var @data=(%num_features)f32
pnnx.Attribute          op_weight   0 1 weight @data
pnnx.Attribute          op_bias     0 1 bias @data
F.batch_norm            op_0        5 1 input running_mean running_var weight bias out eps=%eps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.BatchNorm1d          batchnorm   1 1 input out num_features=%num_features eps=%eps affine=True @running_mean=%op_mean.data @running_var=%op_var.data @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        size_t input_rank = matched_operators.at("op_0")->inputs[0]->shape.size();
        return input_rank == 2 || input_rank == 3;
    }
};

class fuse_static_Fbatchnorm_pass_2d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_mean     0 1 running_mean @data=(%num_features)f32
pnnx.Attribute          op_var      0 1 running_var @data=(%num_features)f32
F.batchnorm             op_0        3 1 input running_mean running_var out weight=None bias=None eps=%eps #input=(?,?,?,?)f32
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.BatchNorm2d          batchnorm   1 1 input out num_features=%num_features eps=%eps affine=False @running_mean=%op_mean.data @running_var=%op_var.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_static_Fbatchnorm_pass_2d_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
pnnx.Attribute          op_mean     0 1 running_mean @data=(%num_features)f32
pnnx.Attribute          op_var      0 1 running_var @data=(%num_features)f32
pnnx.Attribute          op_weight   0 1 weight @data
pnnx.Attribute          op_bias     0 1 bias @data
F.batch_norm            op_0        5 1 input running_mean running_var weight bias out eps=%eps #input=(?,?,?,?)f32
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.BatchNorm2d          batchnorm   1 1 input out num_features=%num_features eps=%eps affine=True @running_mean=%op_mean.data @running_var=%op_var.data @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_static_Fbatchnorm_pass_3d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_mean     0 1 running_mean @data=(%num_features)f32
pnnx.Attribute          op_var      0 1 running_var @data=(%num_features)f32
F.batchnorm             op_0        3 1 input running_mean running_var out weight=None bias=None eps=%eps #input=(?,?,?,?,?)f32
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.BatchNorm3d          batchnorm   1 1 input out num_features=%num_features eps=%eps affine=False @running_mean=%op_mean.data @running_var=%op_var.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_static_Fbatchnorm_pass_3d_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
pnnx.Attribute          op_mean     0 1 running_mean @data=(%num_features)f32
pnnx.Attribute          op_var      0 1 running_var @data=(%num_features)f32
pnnx.Attribute          op_weight   0 1 weight @data
pnnx.Attribute          op_bias     0 1 bias @data
F.batch_norm            op_0        5 1 input running_mean running_var weight bias out eps=%eps #input=(?,?,?,?,?)f32
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.BatchNorm3d          batchnorm   1 1 input out num_features=%num_features eps=%eps affine=True @running_mean=%op_mean.data @running_var=%op_var.data @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_Fbatchnorm_no_affine_pass_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
pnnx.Input              mean        0 1 running_mean
pnnx.Input              var         0 1 running_var
pnnx.Attribute          op_weight   0 1 weight @data
pnnx.Attribute          op_bias     0 1 bias @data
F.batch_norm            op_0        5 1 input running_mean running_var weight bias out eps=%eps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Input              mean        0 1 running_mean
pnnx.Input              var         0 1 running_var
F.batch_norm            op_0        3 1 input running_mean running_var out eps=%eps weight=None bias=None
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& /*matched_operators*/, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        auto weight_data = captured_attrs.at("op_weight.data");
        auto bias_data = captured_attrs.at("op_bias.data");

        std::vector<float> weight_data_fp32 = weight_data.get_float32_data();
        std::vector<float> bias_data_fp32 = bias_data.get_float32_data();

        for (auto w : weight_data_fp32)
        {
            if (w != 1.f)
                return false;
        }
        for (auto b : bias_data_fp32)
        {
            if (b != 0.f)
                return false;
        }

        return true;
    }
};

void fuse_static_batchnorm(Graph& graph)
{
    fuse_static_Fbatchnorm_pass_1d a;
    fuse_static_Fbatchnorm_pass_2d b;
    fuse_static_Fbatchnorm_pass_3d c;
    fuse_static_Fbatchnorm_pass_1d_1 a1;
    fuse_static_Fbatchnorm_pass_2d_1 b1;
    fuse_static_Fbatchnorm_pass_3d_1 c1;
    fuse_Fbatchnorm_no_affine_pass_onnx z;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
    pnnx_graph_rewrite(graph, &c, opindex);
    pnnx_graph_rewrite(graph, &a1, opindex);
    pnnx_graph_rewrite(graph, &b1, opindex);
    pnnx_graph_rewrite(graph, &c1, opindex);
    pnnx_graph_rewrite(graph, &z, opindex);
}

} // namespace pnnx
