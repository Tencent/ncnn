// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

#include <algorithm>

namespace pnnx {

namespace ncnn {

// ncnn has no repeat_interleave layer, decompose the op into existing layers:
//
//   channel axis : Tile + ShuffleChannel                          (2 layers)
//   other axes   : Permute + Tile + ShuffleChannel + Permute       (dims 3 or 4)
//   fallback     : Slice + Tile + Concat                           (any axis, per element repeats)
//   dim omitted  : Reshape + Tile + Reshape, or flatten and use the fallback
//
// the fixed layer count cases use GraphRewriterPass, the fallback is expanded manually

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

// map the torch dim to the ncnn layer axis, the remaining dims keep the torch order
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

    // repeating along the batch axis cannot be expressed with a layer axis param
    if (batch_axis != 233 && dim == batch_axis)
        return false;

    axis = dim;
    if (batch_axis != 233 && axis > batch_axis)
        axis -= 1;

    mat_dims = torch_rank - (batch_axis != 233 ? 1 : 0);
    axis_length = in->shape[dim];

    return mat_dims >= 1 && mat_dims <= 4 && axis_length > 0;
}

// repeats must be a positive constant, a scalar or a 1d per element array
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

// repeats may also be a constant tensor (pnnx.Attribute)
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

// all repeats equal to 1 makes the whole op an identity
static bool all_repeats_one(const std::vector<int>& repeats)
{
    for (size_t i = 0; i < repeats.size(); i++)
    {
        if (repeats[i] != 1)
            return false;
    }
    return true;
}

// permutations that move the target axis into the Mat channel slot, since Tile and
// ShuffleChannel only work on the channel axis (see src/layer/permute.cpp for the order_type
// enum), order_out is the inverse that restores the axis order:
//
//   dims==3 (w h c)  : axis 1 -> 2/2,  axis 2 -> 4/3
//   dims==4 (w h d c): axis 1 -> 6/6,  axis 2 -> 12/8,  axis 3 -> 18/9
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

// dim omitted: torch flattens first, so reshape to (c=T,d=1,h=1,w=1), copy r times along d
// (the same as per element repeats) and reshape back, both params are constants
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

        // flatten and copy along d, the intermediate blobs are 4d
        if (!in->shape.empty())
        {
            const int total = total_shape(in->shape);
            mid->shape = std::vector<int>{total, 1, 1, 1};
            mid2->shape = std::vector<int>{total, repeats, 1, 1};
        }

        // there is no batch axis after the flatten
        mid->params["__ncnn_batch_axis"] = 233;
        mid2->params["__ncnn_batch_axis"] = 233;
        out->params["__ncnn_batch_axis"] = 233;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_repeat_interleave_nodim, 20)

// dim omitted + per element repeats: flatten to 1d first and let the fallback do the rest
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

        // after the flatten the repeats length must match the element count
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

// target axis is the channel axis: Tile copies the channel block, ShuffleChannel interleaves it
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

        // the Mat needs a channel axis and a static ShuffleChannel group
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

        // Tile gives [c0,c1,c0,c1,...], ShuffleChannel(reverse) interleaves it into [c0,c0,c1,c1,...]
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

// other axes: permute the axis into the channel slot, reuse Tile + ShuffleChannel, permute back
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

        // the pair is validated by match()
        int order_in = 0;
        int order_out = 0;
        get_permute_orders(mat_dims, axis, order_in, order_out);

        ops.at("p0")->params["0"] = order_in;
        ops.at("tile")->params["0"] = 0;
        ops.at("tile")->params["1"] = repeats;
        ops.at("sc")->params["0"] = axis_length;
        ops.at("sc")->params["1"] = 1;
        ops.at("p1")->params["0"] = order_out;

        // the axis order is permuted so the blob shape is left empty (same as torch.roll)
        propagate_ncnn_batch_axis(in, ops.at("p0")->outputs[0]);
        propagate_ncnn_batch_axis(in, ops.at("tile")->outputs[0]);
        propagate_ncnn_batch_axis(in, ops.at("sc")->outputs[0]);
        propagate_ncnn_batch_axis(in, ops.at("p1")->outputs[0]);
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_repeat_interleave_permute, 20)

// an axis length of 1 or all repeats equal to 1 is a plain block copy, a single Tile covers both
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

// fallback: slice one element of the axis, repeat it r times and concat the parts. Built by hand
// since the layer count is dynamic and a replacement graph re-matching itself corrupts new_ops
void expand_repeat_interleave(Graph& graph)
{
    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        Operator* op = graph.ops[i];
        if (op->type != "torch.repeat_interleave")
            continue;

        // the dim omitted case is handled by torch_repeat_interleave_nodim*
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

        // an axis length of 1 and all-ones repeats went to the trivial pass
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
                // slice out the first element and keep the rest for the next iteration
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

            // repeat the sliced element r times along this axis
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

        // concatenate the parts in order
        Operator* concat = graph.new_operator_before("Concat", op->name + "_concat", op);
        concat->params["0"] = axis;

        for (int k = 0; k < axis_length; k++)
        {
            concat->inputs.push_back(parts[k]);
            parts[k]->consumers.push_back(concat);
        }

        concat->outputs.push_back(op->outputs[0]);
        op->outputs[0]->producer = concat;

        // drop the reverse references of the original op on all its inputs
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
