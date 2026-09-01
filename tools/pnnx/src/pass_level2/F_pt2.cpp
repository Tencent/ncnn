// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// pt2 路径的形态归一分支(住这里,渐进铺开)。
//
// 默认值缺省形态已由离线静态表解决:torch.export 会省略等于默认值的实参,
// builder(load_pt2)按 src/aten_defaults_table.h 把省略实参补全为完整
// schema 形态,使 pt2 图与 torchscript 图同构,torch_cat / torch_flatten /
// torch_stack / F_linear / F_conv2d_1 等既有 ts 形态分支零改动直接消费。
//
// 本文件收留"无 ts 同文件兄弟"的 pt2 形态差异:torch.export 与 torchscript
// 对同一算子形态不同的场景(分解形态、overload 变体、多输出结构等)。

#include "pass_level2.h"

#include "utils.h"

#include <string.h>

namespace pnnx {

// pt2 形态分支:torch.ones_like 常量折叠。
// torchscript 侧 pass_level0 用 libtorch 实跑模型,把"值不依赖输入"的子图
// (如 maximum(z, ones_like(z)+0.5) 中的 ones_like(z)+0.5)折成常量;pt2
// 路径零 libtorch,而 ones_like 的值语义恒为全 1,可静态折叠:ones_like +
// add(标量) → 全 (1 + alpha*other) 的 pnnx.Attribute,使后续 fuse_expression
// 的表达式形态与 torchscript 侧一致。仅匹配"直接被 add(标量常量) 消费"的
// 形态(现行语料的唯一形态);other 为张量等不满足条件的形态不匹配,保持
// 原样显式失败。
class F_pt2_fold_ones_like : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        // 匹配 torch_ones_like(pass_level2,priority 20)归一后的形态;
        // ones_like 的可省实参已由该 pass 与默认值表消化
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
torch.ones_like         op_0        1 1 input ones_out dtype=%ones_dtype
prim::Constant          op_c        0 1 other value=%other
prim::Constant          op_a        0 1 alpha value=%alpha
aten::add               op_1        3 1 ones_out other alpha out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "pnnx.Attribute";
    }

    const char* name_str() const
    {
        // 对齐 torchscript 侧 fold_constants 的产物命名(normalize 后比对)
        return "pnnx_fold";
    }

    bool match(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        // other 为标量(float/int),alpha 为数值;张量 other 不折叠
        const Parameter& other = captured_params.at("other");
        const Parameter& alpha = captured_params.at("alpha");
        if (other.type != 2 && other.type != 3)
            return false;
        if (alpha.type != 2 && alpha.type != 3)
            return false;
        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params,
               const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Parameter& other = captured_params.at("other");
        const Parameter& alpha = captured_params.at("alpha");

        const float scalar_other = (other.type == 2) ? (float)other.i : other.f;
        const float scalar_alpha = (alpha.type == 2) ? (float)alpha.i : alpha.f;
        const float folded_value = 1.f + scalar_alpha * scalar_other;

        // op->outputs[0] = 原 add 输出 operand,shape/dtype 来自文件元数据
        const Operand* out = op->outputs[0];
        if (out->type != 1 || out->shape.empty())
            return; // 非 f32 或 shape 缺失:保留空 Attribute,后续显式失败

        Attribute attr;
        attr.type = 1;
        attr.shape = out->shape;

        size_t elem_count = 1;
        for (size_t i = 0; i < attr.shape.size(); i++)
            elem_count *= (size_t)attr.shape[i];

        attr.data.resize(elem_count * sizeof(float));
        float* p = (float*)attr.data.data();
        for (size_t i = 0; i < elem_count; i++)
            p[i] = folded_value;

        op->attrs["data"] = attr;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_fold_ones_like, 90)

// pt2 形态分支:weight_norm 参数化权重。
// torch.export 保留 aten::_weight_norm(v, g, dim) 节点;torchscript 侧在
// level1 模块转换(nn_Conv*.cpp)时已用 utils.cpp 的 apply_weight_norm 把
// 折算权重并入 @weight。此处把 v/g 均为 pnnx.Attribute 的 dim=0 形态折成
// pnnx.Attribute(同一 float 实现,权重字节与 ts 一致),使后续
// fuse_static_conv / fuse_static_convtranspose / fuse_static_linear(level5)
// 与 ts 同构;原始 v/g Attribute 失去消费者后由 dead_code_elimination 清理。
// v/g 非常量权重等不满足条件的形态不匹配,留在图内显式失败(不硬凑)。
class F_pt2_weight_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Attribute          op_v        0 1 v @data
pnnx.Attribute          op_g        0 1 g @data
prim::Constant          op_dim      0 1 dim value=%dim
aten::_weight_norm      op_0        3 1 v g dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 v
pnnx.Input              input_1     0 1 g
pnnx.Input              input_2     0 1 dim
pnnx.Attribute          weight      0 1 out @data=(1)f32
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "pnnx.Attribute";
    }

    bool match(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        // 与 level1 模块转换一致只折沿 axis0 的形态;dim 在 schema 里是
        // float(export 物化为 as_float),接受 int/float 两种编码
        const Parameter& dim = captured_params.at("dim");
        const float dim_value = (dim.type == 2) ? (float)dim.i : ((dim.type == 3) ? dim.f : -1.f);
        if (dim_value != 0.f)
            return false;

        const Attribute& attr_v = captured_attrs.at("op_v.data");
        const Attribute& attr_g = captured_attrs.at("op_g.data");

        if (attr_v.type != 1 || attr_g.type != 1)
            return false;

        if (attr_v.shape.empty())
            return false;

        const int dim0 = attr_v.shape[0];
        if (attr_g.get_float32_data().size() != (size_t)dim0)
            return false;

        return true;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const Attribute& attr_v = captured_attrs.at("op_v.data");
        const Attribute& attr_g = captured_attrs.at("op_g.data");

        std::vector<float> weight = attr_v.get_float32_data();
        const std::vector<float>& weight_g = attr_g.get_float32_data();

        const int dim0 = attr_v.shape[0];
        const int size = (int)(weight.size() / dim0);

        apply_weight_norm(weight, weight_g, dim0, size);

        Operator* op_weight = ops.at("weight");
        op_weight->attrs["data"] = Attribute();
        op_weight->attrs["data"].type = attr_v.type;
        op_weight->attrs["data"].shape = attr_v.shape;
        op_weight->attrs["data"].data.resize(weight.size() * sizeof(float));
        memcpy(op_weight->attrs["data"].data.data(), weight.data(), weight.size() * sizeof(float));
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_weight_norm, 130)

// pt2 形态分支:adaptive pool 的 output_size 缺省维还原。
// torch.export 把 output_size 中的 None 实例化成输入空间维尺寸((None,3) →
// 单个 INTS 常量 (24,3)),torchscript 侧保留 None(转换器对 None 写 -233 哨兵)。
// 把"等于输入对应空间维"的元素还原为 0(pass_ncnn 对 0 同样写 -233),两路
// 编码一致;恒等池化语义不变。仅匹配 prim::Constant 形态,ts 的 ListConstruct
// 形态不受影响。需早于 F_adaptive_*(priority 120)消费 aten 原形态。
class F_pt2_adaptive_pool_base : public GraphRewriterPass
{
public:
    // 替换图与匹配图同构,靠"确会发生改写才匹配"终止重写循环:
    // 含 0 元素(已改写,torch 语义 output_size 非 0)或没有元素等于
    // 输入对应空间维(无需改写)时不再匹配
    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Parameter& osz = captured_params.at("output_size");
        if (osz.type != 5)
            return false;

        const std::vector<int>& ishape = matched_operators.at("op_0")->inputs[0]->shape;
        if (ishape.empty())
            return false;

        const int k = (int)osz.ai.size();
        for (int i = 0; i < k; i++)
        {
            if (osz.ai[i] == 0)
                return false;

            const int dim_index = (int)ishape.size() - k + i;
            if (dim_index >= 0 && dim_index < (int)ishape.size() && osz.ai[i] == ishape[dim_index])
                return true;
        }

        return false;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        Parameter osz = captured_params.at("output_size");
        const std::vector<int>& ishape = ops.at("op_0")->inputs[0]->shape;

        if (!ishape.empty())
        {
            const int k = (int)osz.ai.size();
            for (int i = 0; i < k; i++)
            {
                const int dim_index = (int)ishape.size() - k + i;
                if (dim_index >= 0 && dim_index < (int)ishape.size() && osz.ai[i] == ishape[dim_index])
                {
                    osz.ai[i] = 0;
                }
            }
        }

        // 无论是否改写都回写(替换图里的 value=(0) 仅为占位)
        ops.at("op_sz")->params["value"] = osz;
    }
};

class F_pt2_adaptive_avg_pool1d : public F_pt2_adaptive_pool_base
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=%output_size
aten::adaptive_avg_pool1d op_0      2 1 input output_size out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=(0)
aten::adaptive_avg_pool1d op_0      2 1 input output_size out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "aten::adaptive_avg_pool1d";
    }
};

class F_pt2_adaptive_avg_pool2d : public F_pt2_adaptive_pool_base
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=%output_size
aten::adaptive_avg_pool2d op_0      2 1 input output_size out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=(0)
aten::adaptive_avg_pool2d op_0      2 1 input output_size out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "aten::adaptive_avg_pool2d";
    }
};

class F_pt2_adaptive_avg_pool3d : public F_pt2_adaptive_pool_base
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=%output_size
aten::adaptive_avg_pool3d op_0      2 1 input output_size out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=(0)
aten::adaptive_avg_pool3d op_0      2 1 input output_size out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "aten::adaptive_avg_pool3d";
    }
};

class F_pt2_adaptive_max_pool1d : public F_pt2_adaptive_pool_base
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 5
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=%output_size
aten::adaptive_max_pool1d op_0      2 2 input output_size out indices
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 5
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=(0)
aten::adaptive_max_pool1d op_0      2 2 input output_size out indices
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "aten::adaptive_max_pool1d";
    }
};

class F_pt2_adaptive_max_pool2d : public F_pt2_adaptive_pool_base
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 5
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=%output_size
aten::adaptive_max_pool2d op_0      2 2 input output_size out indices
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 5
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=(0)
aten::adaptive_max_pool2d op_0      2 2 input output_size out indices
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "aten::adaptive_max_pool2d";
    }
};

class F_pt2_adaptive_max_pool3d : public F_pt2_adaptive_pool_base
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 5
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=%output_size
aten::adaptive_max_pool3d op_0      2 2 input output_size out indices
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 5
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=(0)
aten::adaptive_max_pool3d op_0      2 2 input output_size out indices
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "aten::adaptive_max_pool3d";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_adaptive_avg_pool1d, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_adaptive_avg_pool2d, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_adaptive_avg_pool3d, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_adaptive_max_pool1d, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_adaptive_max_pool2d, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_adaptive_max_pool3d, 110)

// pt2 形态分支:nn.AdaptiveAvgPool* 模块形态(params 化 output_size)。
// builder 的模块转写(load_pt2)把 nn.AdaptiveAvgPool* 的 output_size 折进
// op->params;torch.export 把 output_size 中的 None 实例化成输入空间维尺寸,
// "等于输入对应空间维"还原为 0 的规则与上面的 operand 形态分支一致,此为
// params 形态版本(pass_ncnn 的 nn_AdaptiveAvgPool* 按 0 写 -233 哨兵)。
class F_pt2_nn_adaptive_avg_pool_base : public GraphRewriterPass
{
public:
    // 与 operand 形态分支相同:替换图与匹配图同构,靠"确会发生改写才匹配"
    // 终止重写循环
    bool match(const std::map<std::string, const Operator*>& matched_operators,
               const std::map<std::string, Parameter>& captured_params,
               const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        std::map<std::string, Parameter>::const_iterator it = captured_params.find("output_size");
        if (it == captured_params.end() || it->second.type != 5)
            return false;

        const std::vector<int>& ishape = matched_operators.at("op_0")->inputs[0]->shape;
        if (ishape.empty())
            return false;

        const std::vector<int>& ai = it->second.ai;
        const int k = (int)ai.size();
        for (int i = 0; i < k; i++)
        {
            if (ai[i] == 0)
                return false;

            const int dim_index = (int)ishape.size() - k + i;
            if (dim_index >= 0 && dim_index < (int)ishape.size() && ai[i] == ishape[dim_index])
                return true;
        }

        return false;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params,
               const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        Parameter osz = captured_params.at("output_size");
        const std::vector<int>& ishape = ops.at("op_0")->inputs[0]->shape;

        if (!ishape.empty())
        {
            const int k = (int)osz.ai.size();
            for (int i = 0; i < k; i++)
            {
                const int dim_index = (int)ishape.size() - k + i;
                if (dim_index >= 0 && dim_index < (int)ishape.size() && osz.ai[i] == ishape[dim_index])
                {
                    osz.ai[i] = 0;
                }
            }
        }

        // 无论是否改写都回写(替换图里的 output_size=(0) 仅为占位)
        ops.at("op_0")->params["output_size"] = osz;
    }
};

class F_pt2_nn_adaptive_avg_pool1d : public F_pt2_nn_adaptive_avg_pool_base
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
nn.AdaptiveAvgPool1d    op_0        1 1 input out output_size=%output_size
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
nn.AdaptiveAvgPool1d    op_0        1 1 input out output_size=(0)
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.AdaptiveAvgPool1d";
    }
};

class F_pt2_nn_adaptive_avg_pool2d : public F_pt2_nn_adaptive_avg_pool_base
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
nn.AdaptiveAvgPool2d    op_0        1 1 input out output_size=%output_size
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
nn.AdaptiveAvgPool2d    op_0        1 1 input out output_size=(0)
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.AdaptiveAvgPool2d";
    }
};

class F_pt2_nn_adaptive_avg_pool3d : public F_pt2_nn_adaptive_avg_pool_base
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
nn.AdaptiveAvgPool3d    op_0        1 1 input out output_size=%output_size
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
nn.AdaptiveAvgPool3d    op_0        1 1 input out output_size=(0)
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.AdaptiveAvgPool3d";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_nn_adaptive_avg_pool1d, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_nn_adaptive_avg_pool2d, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_nn_adaptive_avg_pool3d, 110)

} // namespace pnnx