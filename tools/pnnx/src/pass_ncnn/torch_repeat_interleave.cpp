// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

#include <algorithm>

namespace pnnx {

namespace ncnn {

// torch.repeat_interleave 在 ncnn 里没有对应层，这里把它分解成已有算子的组合：
//
//   A 通道轴    Tile + ShuffleChannel                        (2 层，目标轴是通道轴时最优)
//   B 其它轴    Permute + Tile + ShuffleChannel + Permute     (dims==3 或 4)
//   C 兜底      Slice + Tile + Concat                        (任意轴/维，支持逐元素 repeats)
//   退化        Tile(1) / 恒等
//   dim 省略    Reshape 插单位轴 + Tile + Reshape 合并（O(1) 层）或先 flatten 再走 C
//
// A/B/退化/dim 省略这几种层数固定的情形用 GraphRewriterPass 实现；C 的层数随轴长变化，
// 用 expand_repeat_interleave() 手工建图展开。
//
// 为什么要分开：pattern 重写器的替换图里如果又出现被匹配的类型，pass 就会反复匹配自己
// 刚创建的 op，而 pass_level2.cpp 的 new_ops 里仍留着这些指针，op 被下一轮删掉后收尾
// 命名就会 use-after-free。所以可变层数的展开不走重写器，直接建图（同 pass_level5/unroll_rnn_op.cpp）。
//
// 另外注意：pass_ncnn 里这些重写 pass 是**按注册顺序单次扫描**的，注册顺序就是依赖顺序 ——
// 先做 flatten 的规整，再按 A -> B 的优先级匹配，最后才是退化情形。

static int default_ncnn_batch_axis(int batch_index)
{
    return batch_index == 0 ? 0 : 233;
}

static int get_ncnn_batch_axis(const Operand* r)
{
    if (r->params.find("__ncnn_batch_axis") != r->params.end())
        return r->params.at("__ncnn_batch_axis").i;

    if (r->params.find("__batch_index") != r->params.end())
        return default_ncnn_batch_axis(r->params.at("__batch_index").i);

    return 233;
}

static void propagate_ncnn_batch_axis(const Operand* from, Operand* to)
{
    if (from->params.find("__ncnn_batch_axis") != from->params.end())
        to->params["__ncnn_batch_axis"] = from->params.at("__ncnn_batch_axis").i;

    if (from->params.find("__batch_index") != from->params.end())
        to->params["__batch_index"] = from->params.at("__batch_index").i;
}

// 把 torch 的 dim 换算成 ncnn 层的 axis 参数：去掉 batch 轴后，剩下的维按 torch 顺序编号
static bool resolve_ncnn_axis(const Operand* in, int dim, int& axis, int& mat_dims, int& axis_length)
{
    if (in->shape.empty())
        return false;

    const int torch_rank = (int)in->shape.size();
    if (dim < 0)
        dim += torch_rank;
    if (dim < 0 || dim >= torch_rank)
        return false;

    const int batch_axis = get_ncnn_batch_axis(in);

    // 沿 batch 轴重复没法用 ncnn 层的 axis 参数表达
    if (batch_axis != 233 && dim == batch_axis)
        return false;

    axis = dim;
    if (batch_axis != 233 && axis > batch_axis)
        axis -= 1;

    mat_dims = torch_rank - (batch_axis != 233 ? 1 : 0);
    axis_length = in->shape[dim];

    return mat_dims >= 1 && mat_dims <= 4 && axis_length > 0;
}

// repeats 必须是正整数的常量：标量（统一重复）或一维数组（逐元素重复）
static bool get_repeats(const Parameter& p, std::vector<int>& repeats)
{
    if (p.type == 2)
    {
        if (p.i < 1)
            return false;

        repeats.assign(1, p.i);
        return true;
    }

    if (p.type == 5)
    {
        if (p.ai.empty())
            return false;

        for (size_t i = 0; i < p.ai.size(); i++)
        {
            if (p.ai[i] < 1)
                return false;
        }

        repeats = p.ai;
        return true;
    }

    return false;
}

static bool get_repeats_from_attribute(const Attribute& a, std::vector<int>& repeats)
{
    int size = 1;
    for (size_t i = 0; i < a.shape.size(); i++)
    {
        size *= a.shape[i];
    }

    if (size < 1)
        return false;

    repeats.resize(size);

    if (a.type == 4)
    {
        const int* p = (const int*)a.data.data();
        for (int i = 0; i < size; i++)
        {
            repeats[i] = p[i];
        }
    }
    else if (a.type == 5)
    {
        const int64_t* p = (const int64_t*)a.data.data();
        for (int i = 0; i < size; i++)
        {
            repeats[i] = (int)p[i];
        }
    }
    else
    {
        return false;
    }

    for (int i = 0; i < size; i++)
    {
        if (repeats[i] < 1)
            return false;
    }

    return true;
}

// repeats 也可能是常量张量（pnnx.Attribute）
static bool get_repeats_from_operand(const Operand* r, std::vector<int>& repeats)
{
    const Operator* op = r->producer;
    if (!op || op->type != "pnnx.Attribute" || op->attrs.size() != 1)
        return false;

    return get_repeats_from_attribute(op->attrs.begin()->second, repeats);
}

static std::vector<int> shape_with_axis_length(const std::vector<int>& shape, int dim, int length)
{
    std::vector<int> s = shape;
    if (dim >= 0 && dim < (int)s.size())
    {
        s[dim] = length;
    }
    return s;
}

static int total_shape(const std::vector<int>& shape)
{
    int total = 1;
    for (size_t i = 0; i < shape.size(); i++)
    {
        total *= shape[i];
    }
    return total;
}

// repeats 全为 1 时整个算子就是恒等
static bool all_repeats_one(const std::vector<int>& repeats)
{
    for (size_t i = 0; i < repeats.size(); i++)
    {
        if (repeats[i] != 1)
            return false;
    }
    return true;
}

// Permute 的 order_type 枚举见 src/layer/permute.cpp。
// 这里给出把目标轴搬到 Mat 通道槽（Tile/ShuffleChannel 只能作用在通道轴）所用的置换对：
//
//   dims==3 (w h c): 轴 1=h -> order 2  轴 2=w -> order 4
//   dims==4 (w h d c): 轴 1=d -> order 6  轴 2=h -> order 12  轴 3=w -> order 18
//
// 返回的 order_out 是还原轴序的逆置换（已在 ncnn 上逐一对拍验证）。
static bool get_permute_orders(int mat_dims, int axis, int& order_in, int& order_out)
{
    if (mat_dims == 3)
    {
        if (axis == 1)
        {
            order_in = 2;
            order_out = 2;
            return true;
        }
        if (axis == 2)
        {
            order_in = 4;
            order_out = 3;
            return true;
        }
        return false;
    }

    if (mat_dims == 4)
    {
        if (axis == 1)
        {
            order_in = 6;
            order_out = 6;
            return true;
        }
        if (axis == 2)
        {
            order_in = 12;
            order_out = 8;
            return true;
        }
        if (axis == 3)
        {
            order_in = 18;
            order_out = 9;
            return true;
        }
        return false;
    }

    return false;
}

// dim 省略时 torch 会先把输入 flatten 成一维再做重复，
// 先用 Reshape 拆出 (c=T,d=1,h=1,w=1)，沿 d 复制 r 份等价于逐元素重复，最后再合并回一维。
// 这个分解不依赖具体尺寸（Tile 的倍数与 Reshape 的目标形状都是常量）
class torch_repeat_interleave_nodim : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.repeat_interleave op_0        1 1 input out dim=%dim repeats=%repeats
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input    0 1 input
Reshape                 reshape  1 1 input mid 0=1 1=1 11=1 2=-1
Tile                    tile     1 1 mid mid2 0=1
Reshape                 reshape2 1 1 mid2 out 0=-1
pnnx.Output             output   1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& /*matched_operators*/, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        if (captured_params.at("dim").type != 0)
            return false;

        std::vector<int> repeats;
        if (!get_repeats(captured_params.at("repeats"), repeats) || repeats.size() != 1)
            return false;

        return true;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int repeats = captured_params.at("repeats").i;

        Operand* in = ops.at("reshape")->inputs[0];

        ops.at("tile")->params["1"] = repeats;

        Operand* mid = ops.at("reshape")->outputs[0];
        Operand* mid2 = ops.at("tile")->outputs[0];
        Operand* out = ops.at("reshape2")->outputs[0];

        // flatten + 沿 d 轴复制，中间 blob 是 4 维的
        if (!in->shape.empty())
        {
            const int total = total_shape(in->shape);
            mid->shape = std::vector<int>{total, 1, 1, 1};
            mid2->shape = std::vector<int>{total, repeats, 1, 1};
        }

        // flatten 之后就没有 batch 轴了
        mid->params["__ncnn_batch_axis"] = 233;
        mid2->params["__ncnn_batch_axis"] = 233;
        out->params["__ncnn_batch_axis"] = 233;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_repeat_interleave_nodim, 20)

// dim 省略 + 逐元素 repeats：同样先 flatten 成一维，剩下的交给 expand_repeat_interleave()
class torch_repeat_interleave_nodim_attribute : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          op_repeats  0 1 repeats @data
torch.repeat_interleave op_0        2 1 input repeats out dim=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input    0 1 input
Reshape                 reshape  1 1 input mid 0=-1
torch.repeat_interleave op_1     1 1 mid out dim=0
pnnx.Output             output   1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        if (captured_params.at("dim").type != 0)
            return false;

        std::vector<int> repeats;
        if (!get_repeats_from_attribute(captured_attrs.at("op_repeats.data"), repeats))
            return false;

        const Operand* in = matched_operators.at("op_0")->inputs[0];
        if (in->shape.empty())
            return false;

        // flatten 之后 repeats 的长度必须等于元素总数
        return (int)repeats.size() == total_shape(in->shape);
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        std::vector<int> repeats;
        get_repeats_from_attribute(captured_attrs.at("op_repeats.data"), repeats);

        Operand* in = ops.at("reshape")->inputs[0];

        ops.at("op_1")->params["repeats"] = repeats;

        Operand* mid = ops.at("reshape")->outputs[0];
        Operand* out = ops.at("op_1")->outputs[0];

        mid->shape = std::vector<int>{total_shape(in->shape)};
        mid->params["__ncnn_batch_axis"] = 233;
        out->params["__ncnn_batch_axis"] = 233;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_repeat_interleave_nodim_attribute, 20)

// A 路径：目标轴就是通道轴时，Tile 沿通道整块复制，再用 ShuffleChannel 交错
class torch_repeat_interleave_channel : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.repeat_interleave op_0        1 1 input out dim=%dim repeats=%repeats
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input      input   0 1 input
Tile            tile    1 1 input mid
ShuffleChannel  sc      1 1 mid out
pnnx.Output     output  1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        if (captured_params.at("dim").type != 2)
            return false;

        std::vector<int> repeats;
        if (!get_repeats(captured_params.at("repeats"), repeats) || repeats.size() != 1)
            return false;

        int axis, mat_dims, axis_length;
        if (!resolve_ncnn_axis(matched_operators.at("op_0")->inputs[0], captured_params.at("dim").i, axis, mat_dims, axis_length))
            return false;

        // 需要 Mat 至少有通道维，且 ShuffleChannel 的 group 必须是静态的；重复 1 次留给退化分支
        return axis == 0 && mat_dims >= 3 && axis_length >= 2 && repeats[0] >= 2;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int dim = captured_params.at("dim").i;
        const int repeats = captured_params.at("repeats").i;

        Operand* in = ops.at("tile")->inputs[0];

        int axis, mat_dims, axis_length;
        resolve_ncnn_axis(in, dim, axis, mat_dims, axis_length);

        // Tile 得到 [c0,c1,...,c0,c1,...]，ShuffleChannel(reverse) 再交错成 [c0,c0,c1,c1,...]
        ops.at("tile")->params["0"] = axis;
        ops.at("tile")->params["1"] = repeats;
        ops.at("sc")->params["0"] = axis_length;
        ops.at("sc")->params["1"] = 1;

        Operand* mid = ops.at("tile")->outputs[0];
        Operand* out = ops.at("sc")->outputs[0];

        mid->shape = shape_with_axis_length(in->shape, dim, axis_length * repeats);
        propagate_ncnn_batch_axis(in, mid);
        propagate_ncnn_batch_axis(in, out);
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_repeat_interleave_channel, 20)

// B 路径：先把目标轴置换到通道位，复用 A 的组合，再置换回来（dims==3 或 4）
class torch_repeat_interleave_permute : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.repeat_interleave op_0        1 1 input out dim=%dim repeats=%repeats
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input      input   0 1 input
Permute         p0      1 1 input mid0
Tile            tile    1 1 mid0 mid1
ShuffleChannel  sc      1 1 mid1 mid2
Permute         p1      1 1 mid2 out
pnnx.Output     output  1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        if (captured_params.at("dim").type != 2)
            return false;

        std::vector<int> repeats;
        if (!get_repeats(captured_params.at("repeats"), repeats) || repeats.size() != 1)
            return false;

        int axis, mat_dims, axis_length;
        if (!resolve_ncnn_axis(matched_operators.at("op_0")->inputs[0], captured_params.at("dim").i, axis, mat_dims, axis_length))
            return false;

        if (axis_length < 2 || repeats[0] < 2)
            return false;

        int order_in, order_out;
        return get_permute_orders(mat_dims, axis, order_in, order_out);
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int dim = captured_params.at("dim").i;
        const int repeats = captured_params.at("repeats").i;

        Operand* in = ops.at("p0")->inputs[0];

        int axis, mat_dims, axis_length;
        resolve_ncnn_axis(in, dim, axis, mat_dims, axis_length);

        // match() 已经校验过，这里必定能取到置换对
        int order_in = 0;
        int order_out = 0;
        get_permute_orders(mat_dims, axis, order_in, order_out);

        ops.at("p0")->params["0"] = order_in;
        ops.at("tile")->params["0"] = 0;
        ops.at("tile")->params["1"] = repeats;
        ops.at("sc")->params["0"] = axis_length;
        ops.at("sc")->params["1"] = 1;
        ops.at("p1")->params["0"] = order_out;

        // 中间几个 blob 的轴序被置换过，shape 留空（与 torch.roll 的做法一致），只传 batch 轴信息
        propagate_ncnn_batch_axis(in, ops.at("p0")->outputs[0]);
        propagate_ncnn_batch_axis(in, ops.at("tile")->outputs[0]);
        propagate_ncnn_batch_axis(in, ops.at("sc")->outputs[0]);
        propagate_ncnn_batch_axis(in, ops.at("p1")->outputs[0]);
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_repeat_interleave_permute, 20)

// 退化情形：目标轴长度为 1 时整块复制与逐元素重复等价；repeats 全为 1 时算子就是恒等。
// 两种都退化成单个 Tile（tiles 为 1 时 Tile 是恒等）
class torch_repeat_interleave_trivial : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.repeat_interleave op_0        1 1 input out dim=%dim repeats=%repeats
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tile";
    }

    const char* name_str() const
    {
        return "repeat_interleave";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        if (captured_params.at("dim").type != 2)
            return false;

        std::vector<int> repeats;
        if (!get_repeats(captured_params.at("repeats"), repeats))
            return false;

        int axis, mat_dims, axis_length;
        if (!resolve_ncnn_axis(matched_operators.at("op_0")->inputs[0], captured_params.at("dim").i, axis, mat_dims, axis_length))
            return false;

        return axis_length == 1 || all_repeats_one(repeats);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        std::vector<int> repeats;
        get_repeats(captured_params.at("repeats"), repeats);

        int axis, mat_dims, axis_length;
        resolve_ncnn_axis(op->inputs[0], captured_params.at("dim").i, axis, mat_dims, axis_length);

        op->params["0"] = axis;
        op->params["1"] = repeats[0];
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_repeat_interleave_trivial, 20)

// C 路径：层数随轴长变化，手工建图展开成 Slice + Tile + Concat
// 之所以不做成 pattern 替换：替换图里再出现同类型 op 会让 pass 反复匹配自己刚创建的 op，
// 而 pass_level2.cpp 的 new_ops 里仍留着这些指针（下一轮删掉后收尾命名就 use-after-free）。
// 这里每次切出轴上的一个元素复制 r 份，最后把各段拼接起来。
void expand_repeat_interleave(Graph& graph)
{
    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        Operator* op = graph.ops[i];
        if (op->type != "torch.repeat_interleave")
            continue;

        // dim 省略的情形由 torch_repeat_interleave_nodim* 处理
        if (op->params.find("dim") == op->params.end() || op->params.at("dim").type != 2)
            continue;

        std::vector<int> repeats;
        if (op->params.find("repeats") != op->params.end())
        {
            if (!get_repeats(op->params.at("repeats"), repeats))
                continue;
        }
        else if (op->inputs.size() == 2)
        {
            if (!get_repeats_from_operand(op->inputs[1], repeats))
                continue;
        }
        else
        {
            continue;
        }

        Operand* in = op->inputs[0];

        int axis, mat_dims, axis_length;
        if (!resolve_ncnn_axis(in, op->params.at("dim").i, axis, mat_dims, axis_length))
            continue;

        // 轴长度为 1 或 repeats 全为 1 的情形已经由 torch_repeat_interleave_trivial 处理
        if (axis_length < 2 || all_repeats_one(repeats))
            continue;

        if (repeats.size() != 1 && (int)repeats.size() != axis_length)
            continue;

        std::vector<Operand*> parts(axis_length);
        Operand* cur = in;
        int cur_length = axis_length;

        for (int k = 0; k < axis_length; k++)
        {
            const int r = repeats.size() == 1 ? repeats[0] : repeats[k];

            Operand* elem = cur;
            if (k != axis_length - 1)
            {
                // 切出第一个元素，剩下的留给下一轮
                Operator* slice = graph.new_operator_before("Slice", op->name + "_slice_" + std::to_string(k), op);
                slice->params["0"] = std::vector<int>{1, cur_length - 1};
                slice->params["1"] = axis;

                slice->inputs.push_back(cur);
                cur->remove_consumer(op);
                cur->consumers.push_back(slice);

                Operand* elem_operand = graph.new_operand(op->name + "_elem_" + std::to_string(k));
                elem_operand->producer = slice;
                elem_operand->shape = shape_with_axis_length(in->shape, op->params.at("dim").i, 1);
                propagate_ncnn_batch_axis(in, elem_operand);
                slice->outputs.push_back(elem_operand);

                Operand* rest_operand = graph.new_operand(op->name + "_rest_" + std::to_string(k));
                rest_operand->producer = slice;
                rest_operand->shape = shape_with_axis_length(in->shape, op->params.at("dim").i, cur_length - 1);
                propagate_ncnn_batch_axis(in, rest_operand);
                slice->outputs.push_back(rest_operand);

                elem = elem_operand;
                cur = rest_operand;
                cur_length -= 1;
            }

            // 把切出来的元素沿该轴复制 r 份
            Operator* tile = graph.new_operator_before("Tile", op->name + "_tile_" + std::to_string(k), op);
            tile->params["0"] = axis;
            tile->params["1"] = r;

            tile->inputs.push_back(elem);
            elem->consumers.push_back(tile);

            Operand* part_operand = graph.new_operand(op->name + "_part_" + std::to_string(k));
            part_operand->producer = tile;
            part_operand->shape = shape_with_axis_length(in->shape, op->params.at("dim").i, r);
            propagate_ncnn_batch_axis(in, part_operand);
            tile->outputs.push_back(part_operand);

            parts[k] = part_operand;
        }

        // 按顺序拼回去
        Operator* concat = graph.new_operator_before("Concat", op->name + "_concat", op);
        concat->params["0"] = axis;

        for (int k = 0; k < axis_length; k++)
        {
            concat->inputs.push_back(parts[k]);
            parts[k]->consumers.push_back(concat);
        }

        concat->outputs.push_back(op->outputs[0]);
        op->outputs[0]->producer = concat;

        // 摘掉原 op 在所有输入上的反向引用（repeats 是常量张量时，那个 pnnx.Attribute
        // 也会变成死代码，由后面的 dead_code_elimination 回收）
        for (size_t j = 0; j < op->inputs.size(); j++)
        {
            op->inputs[j]->remove_consumer(op);
        }

        op->inputs.clear();
        op->outputs.clear();

        graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op));

        delete op;

        if (i > 0)
            i--;
    }
}

} // namespace ncnn

} // namespace pnnx
