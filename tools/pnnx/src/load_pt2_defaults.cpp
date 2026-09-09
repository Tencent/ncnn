// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// default-kwargs restoration for the pt2 loader (split out of
// load_exportedprogram.cpp).
//
// torch.export graph JSON omits schema default arguments; append_default_kwargs
// fills them back in by parameter name (add_const) so the translated ops carry
// every input the pass_level2 patterns expect. reorder_inputs restores the
// canonical schema order when the exported overload omitted middle defaults.

#include "load_pt2_defaults.h"

#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <climits>
#include <string>
#include <vector>

#include "ir.h"

namespace pnnx {

// create a prim::Constant operator and wire it as an input of the consumer
// note: must be inserted before the consumer so that pass_level3
// fuse_expression (iterating backwards) handles the consumer first while the
// constant is still a prim::Constant and can be inlined correctly; otherwise
// the constant is fused into a pnnx.Expression first and the consumer's expr
// ends up with dangling @N references
void new_constant(Graph& g, Operator* consumer, const Parameter& value, int& constant_index)
{
    char name[32];
    snprintf(name, 32, "pnnx_constant_%d", constant_index++);

    Operator* op = g.new_operator_before("prim::Constant", name, consumer);
    op->params["value"] = value;

    Operand* r = g.new_operand(name);
    r->producer = op;
    op->outputs.push_back(r);

    r->consumers.push_back(consumer);
    consumer->inputs.push_back(r);
}

// append default scalar inputs for aten operators that omitted default kwargs
// dynamo omits schema default arguments; fill them by parameter name here,
// keeping the input order consistent with the pass_level2 patterns (omitted
// ones are trailing defaults, so appending keeps the order)
// shared context + helpers for the per-op default-fill handlers below.
// add_const appends a default prim::Constant input (keeping op->inputnames in
// sync); reorder_inputs restores a canonical order when the exported overload
// omitted middle defaults and blindly appending would misalign the
// position-matched level-2 patterns.
struct DefaultsCtx
{
    Graph& g;
    Operator* op;
    const std::string& type;
    const std::vector<std::string>& inputnames;
    int& constant_index;

    void add_const(const std::string& name, const Parameter& value)
    {
        new_constant(g, op, value, constant_index);
        op->inputnames.push_back(name);
    }

    void reorder_inputs(const std::vector<std::string>& order)
    {
        // guard against name/operand skew (e.g. a scalar arg the loader has no
        // branch for leaves a dangling inputname with no matching operand);
        // never index past op->inputs on a longer inputnames list
        if (op->inputs.size() != op->inputnames.size())
            return;

        std::vector<Operand*> new_inputs;
        std::vector<std::string> new_names;
        for (const std::string& nm : order)
        {
            for (size_t k = 0; k < op->inputnames.size(); k++)
            {
                if (op->inputnames[k] == nm)
                {
                    new_inputs.push_back(op->inputs[k]);
                    new_names.push_back(nm);
                    break;
                }
            }
        }
        if (new_names.size() == op->inputnames.size())
        {
            op->inputs = new_inputs;
            op->inputnames = new_names;
        }
    }

    bool has_input_name(const std::string& name) const
    {
        for (size_t i = 0; i < inputnames.size(); i++)
            if (inputnames[i] == name)
                return true;
        return false;
    }

    // find the prim::Constant value of an input by parameter name
    // used for defaults that depend on other params, e.g. stride = kernel_size
    Parameter find_input_value(const std::string& name)
    {
        for (size_t i = 0; i < inputnames.size() && i < op->inputs.size(); i++)
        {
            if (inputnames[i] == name)
            {
                Operator* prod = op->inputs[i]->producer;
                if (prod && prod->type == "prim::Constant" && prod->params.find("value") != prod->params.end())
                    return prod->params["value"];
            }
        }
        return Parameter();
    }
};

typedef void (*DefaultsHandler)(DefaultsCtx&);

static void append_conv_defaults(DefaultsCtx& c)
{
    int dim = 2;
    if (c.type == "aten::conv1d")
        dim = 1;
    else if (c.type == "aten::conv3d")
        dim = 3;

    std::vector<int> ones(dim, 1);
    std::vector<int> zeros(dim, 0);

    if (!c.has_input_name("stride"))
        c.add_const("stride", ones);
    if (!c.has_input_name("padding"))
        c.add_const("padding", zeros);
    if (!c.has_input_name("dilation"))
        c.add_const("dilation", ones);
    if (!c.has_input_name("groups"))
        c.add_const("groups", 1);
}

static void append_batch_norm_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("training"))
        c.add_const("training", false);
    if (!c.has_input_name("momentum"))
        c.add_const("momentum", 0.1f);
    if (!c.has_input_name("eps"))
        c.add_const("eps", 1e-5f);
    if (!c.has_input_name("cudnn_enabled"))
        c.add_const("cudnn_enabled", true);
}

static void append_add_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("alpha"))
        c.add_const("alpha", 1);
}

static void append_max_pool_defaults(DefaultsCtx& c)
{
    int dim = 2;
    if (c.type == "aten::max_pool1d" || c.type == "aten::max_pool1d_with_indices")
        dim = 1;
    else if (c.type == "aten::max_pool3d" || c.type == "aten::max_pool3d_with_indices")
        dim = 3;

    // torch max_pool stride defaults to kernel_size
    Parameter kernel = c.find_input_value("kernel_size");

    if (!c.has_input_name("stride"))
    {
        if (kernel.type == 5)
            c.add_const("stride", kernel.ai);
        else
            c.add_const("stride", std::vector<int>(dim, 1));
    }
    if (!c.has_input_name("padding"))
        c.add_const("padding", std::vector<int>(dim, 0));
    if (!c.has_input_name("dilation"))
        c.add_const("dilation", std::vector<int>(dim, 1));
    if (!c.has_input_name("ceil_mode"))
        c.add_const("ceil_mode", false);
}

static void append_avg_pool_defaults(DefaultsCtx& c)
{
    int dim = 2;
    if (c.type == "aten::avg_pool1d")
        dim = 1;
    else if (c.type == "aten::avg_pool3d")
        dim = 3;

    // torch avg_pool stride defaults to kernel_size
    Parameter kernel = c.find_input_value("kernel_size");

    if (!c.has_input_name("stride"))
    {
        if (kernel.type == 5)
            c.add_const("stride", kernel.ai);
        else
            c.add_const("stride", std::vector<int>(dim, 1));
    }
    if (!c.has_input_name("padding"))
        c.add_const("padding", std::vector<int>(dim, 0));
    if (!c.has_input_name("ceil_mode"))
        c.add_const("ceil_mode", false);
    if (!c.has_input_name("count_include_pad"))
        c.add_const("count_include_pad", true);
    if (c.type != "aten::avg_pool1d")
    {
        if (!c.has_input_name("divisor_override"))
            c.add_const("divisor_override", Parameter());
    }
}

static void append_argmax_defaults(DefaultsCtx& c)
{
    // dim=None means full reduction (torch.argmax(x) without dim)
    if (!c.has_input_name("dim"))
        c.add_const("dim", Parameter());
    if (!c.has_input_name("keepdim"))
        c.add_const("keepdim", false);
}

static void append_sum_mean_defaults(DefaultsCtx& c)
{
    // mean.default / sum full-reduction versions have no keepdim argument
    if (c.has_input_name("dim"))
    {
        if (!c.has_input_name("keepdim"))
            c.add_const("keepdim", false);
        if (!c.has_input_name("dtype"))
            c.add_const("dtype", Parameter());
        // keepdim sits before dtype in the schema; when dtype was already
        // serialized but keepdim omitted, reorder so the level-2 pattern
        // does not read dtype as keepdim
        c.reorder_inputs({"self", "dim", "keepdim", "dtype"});
    }
    else
    {
        // full-reduction overload: self (+ optional dtype) is already in order
        if (!c.has_input_name("dtype"))
            c.add_const("dtype", Parameter());
    }
}

static void append_var_std_defaults(DefaultsCtx& c)
{
    // aten::var/std.dim overloads serialize an unbiased argument; the
    // .correction overload serializes correction. never mix the two families
    // (a 5-input node matches no level-2 pattern), and keep the defaults in
    // canonical schema order so the torch_std/torch_var rewrites match.
    if (c.has_input_name("unbiased"))
    {
        if (!c.has_input_name("keepdim"))
            c.add_const("keepdim", false);
        c.reorder_inputs({"self", "dim", "unbiased", "keepdim"});
    }
    else if (c.has_input_name("correction"))
    {
        if (!c.has_input_name("keepdim"))
            c.add_const("keepdim", false);
        c.reorder_inputs({"self", "dim", "correction", "keepdim"});
    }
    else if (c.has_input_name("dim"))
    {
        // dim overload with the default unbiased/keepdim omitted
        if (!c.has_input_name("unbiased"))
            c.add_const("unbiased", true);
        if (!c.has_input_name("keepdim"))
            c.add_const("keepdim", false);
        c.reorder_inputs({"self", "dim", "unbiased", "keepdim"});
    }
    else
    {
        // reduce-all overload (self only, serialized under .correction)
        if (!c.has_input_name("correction"))
            c.add_const("correction", 1);
        if (!c.has_input_name("keepdim"))
            c.add_const("keepdim", false);
        // the exporter may serialize [self keepdim] when correction was the
        // only omitted middle default (torch.var(x, keepdim=True)); blindly
        // appending correction would yield [self keepdim correction], but the
        // reduce-all torch_var/torch_std patterns expect [self correction
        // keepdim] - restore the canonical order
        c.reorder_inputs({"self", "correction", "keepdim"});
    }
}

static void append_softmax_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("dtype"))
        c.add_const("dtype", Parameter());
}

static void append_pad_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("mode"))
        c.add_const("mode", std::string("constant"));
    if (!c.has_input_name("value"))
        c.add_const("value", Parameter());
}

static void append_to_defaults(DefaultsCtx& c)
{
    // dynamo emits to.dtype / to.device / to.dtype_layout overloads with
    // the trailing defaults omitted; fill them to match the Tensor_to
    // patterns. the dtype_layout overload also carries pin_memory between
    // device and non_blocking.
    if (!c.has_input_name("non_blocking"))
        c.add_const("non_blocking", false);
    if (!c.has_input_name("copy"))
        c.add_const("copy", false);
    if (!c.has_input_name("memory_format"))
        c.add_const("memory_format", Parameter());
    if (c.has_input_name("layout"))
    {
        if (!c.has_input_name("pin_memory"))
            c.add_const("pin_memory", false);
        c.reorder_inputs({"self", "dtype", "layout", "device", "pin_memory", "non_blocking", "copy", "memory_format"});
    }
    else if (c.has_input_name("device"))
    {
        c.reorder_inputs({"self", "device", "dtype", "non_blocking", "copy", "memory_format"});
    }
    else
    {
        c.reorder_inputs({"self", "dtype", "non_blocking", "copy", "memory_format"});
    }
}

static void append_contiguous_defaults(DefaultsCtx& c)
{
    // eliminate_contiguous expects 2 inputs (input + memory_format); dynamo omits memory_format
    if (!c.has_input_name("memory_format"))
        c.add_const("memory_format", Parameter());
}

static void append_slice_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("step"))
        c.add_const("step", 1);
}

static void append_slice_scatter_defaults(DefaultsCtx& c)
{
    // slice_scatter(self, src) omits the dim=0/start=None defaults; the
    // [input src dim start end step] level-2 pattern needs every slot or the
    // node survives as an illegal aten::slice_scatter call
    if (!c.has_input_name("dim"))
        c.add_const("dim", 0);
    if (!c.has_input_name("start"))
        c.add_const("start", Parameter());
    if (!c.has_input_name("end"))
        c.add_const("end", INT_MAX);
    if (!c.has_input_name("step"))
        c.add_const("step", 1);
    c.reorder_inputs({"self", "src", "dim", "start", "end", "step"});
}

static void append_flatten_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("start_dim"))
        c.add_const("start_dim", 0);
    if (!c.has_input_name("end_dim"))
        c.add_const("end_dim", -1);
}

static void append_celu_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("alpha"))
        c.add_const("alpha", 1.0f);
}

static void append_elu_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("alpha"))
        c.add_const("alpha", 1.0f);
    if (!c.has_input_name("scale"))
        c.add_const("scale", 1.0f);
    if (!c.has_input_name("input_scale"))
        c.add_const("input_scale", 1.0f);
}

static void append_hardshrink_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("lambd"))
        c.add_const("lambd", 0.5f);
}

static void append_hardtanh_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("min_val"))
        c.add_const("min_val", -1.0f);
    if (!c.has_input_name("max_val"))
        c.add_const("max_val", 1.0f);
}

static void append_leaky_relu_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("negative_slope"))
        c.add_const("negative_slope", 0.01f);
}

static void append_softplus_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("beta"))
        c.add_const("beta", 1.0f);
    if (!c.has_input_name("threshold"))
        c.add_const("threshold", 20.0f);
}

static void append_softshrink_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("lambd"))
        c.add_const("lambd", 0.5f);
}

static void append_rrelu_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("lower"))
        c.add_const("lower", 0.125f);
    if (!c.has_input_name("upper"))
        c.add_const("upper", 1.0f / 3.0f);
    if (!c.has_input_name("training"))
        c.add_const("training", false);
    if (!c.has_input_name("generator"))
        c.add_const("generator", Parameter());
}

static void append_pairwise_distance_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("p"))
        c.add_const("p", 2);
    if (!c.has_input_name("eps"))
        c.add_const("eps", 1e-6f);
    if (!c.has_input_name("keepdim"))
        c.add_const("keepdim", false);
}

static void append_linear_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("bias"))
        c.add_const("bias", Parameter());
}

static void append_as_strided_defaults(DefaultsCtx& c)
{
    // as_strided(self, size, stride) omits storage_offset=0; append it so
    // the [input size stride storage_offset] level-2 pattern matches
    if (!c.has_input_name("storage_offset"))
        c.add_const("storage_offset", 0);
}

static void append_tril_defaults(DefaultsCtx& c)
{
    // tril(input) omits the diagonal=0 default; append it so the
    // [input diagonal] level-2 pattern matches
    if (!c.has_input_name("diagonal"))
        c.add_const("diagonal", 0);
}

static void append_rms_norm_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("weight"))
        c.add_const("weight", Parameter());
    if (!c.has_input_name("eps"))
        c.add_const("eps", Parameter());
}

static void append_scaled_dot_product_attention_defaults(DefaultsCtx& c)
{
    // dynamo omits intermediate default args and enable_gqa moves up to the
    // attn_mask slot; reorder to match the pattern:
    // query key value attn_mask dropout_p is_causal scale enable_gqa
    static const std::vector<std::string> order = {"query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa"};

    std::vector<Operand*> old_inputs = c.op->inputs;
    std::vector<std::string> old_names = c.op->inputnames;
    c.op->inputs.clear();
    c.op->inputnames.clear();

    for (size_t i = 0; i < order.size(); i++)
    {
        const std::string& name = order[i];

        int found = -1;
        for (size_t j = 0; j < old_names.size(); j++)
        {
            if (old_names[j] == name)
            {
                found = (int)j;
                break;
            }
        }

        if (found != -1)
        {
            c.op->inputs.push_back(old_inputs[found]);
            c.op->inputnames.push_back(name);
        }
        else
        {
            Parameter v;
            if (name == "attn_mask")
                v = Parameter();
            else if (name == "dropout_p")
                v = Parameter(0.0f);
            else if (name == "is_causal")
                v = Parameter(false);
            else if (name == "scale")
                v = Parameter();
            else if (name == "enable_gqa")
                v = Parameter(false);

            new_constant(c.g, c.op, v, c.constant_index);
            c.op->inputnames.push_back(name);
        }
    }
}

static void append_embedding_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("padding_idx"))
        c.add_const("padding_idx", -1);
    if (!c.has_input_name("scale_grad_by_freq"))
        c.add_const("scale_grad_by_freq", false);
    if (!c.has_input_name("sparse"))
        c.add_const("sparse", false);
}

static void append_glu_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("dim"))
        c.add_const("dim", -1);
}

static void append_conv_transpose_defaults(DefaultsCtx& c)
{
    int dim = 2;
    if (c.type == "aten::conv_transpose1d")
        dim = 1;
    else if (c.type == "aten::conv_transpose3d")
        dim = 3;

    if (!c.has_input_name("stride"))
        c.add_const("stride", std::vector<int>(dim, 1));
    if (!c.has_input_name("padding"))
        c.add_const("padding", std::vector<int>(dim, 0));
    if (!c.has_input_name("output_padding"))
        c.add_const("output_padding", std::vector<int>(dim, 0));
    if (!c.has_input_name("groups"))
        c.add_const("groups", 1);
    if (!c.has_input_name("dilation"))
        c.add_const("dilation", std::vector<int>(dim, 1));
}

static void append_amax_defaults(DefaultsCtx& c)
{
    // dim defaults to None (reduce all); when omitted add the null dim slot
    // so [self, dim, keepdim] matches the level-2 pattern instead of
    // leaving a keepdim-only node that no pattern rewrites
    if (!c.has_input_name("dim"))
        c.add_const("dim", Parameter());
    if (!c.has_input_name("keepdim"))
        c.add_const("keepdim", false);
    c.reorder_inputs({"self", "dim", "keepdim"});
}

static void append_max_min_defaults(DefaultsCtx& c)
{
    // max.dim/min.dim omit keepdim when it is False; inputnames has dim but
    // lacks keepdim (max.other/default have no dim and are unaffected)
    if (c.has_input_name("dim") && !c.has_input_name("keepdim"))
        c.add_const("keepdim", false);
}

static void append_logsumexp_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("keepdim"))
        c.add_const("keepdim", false);
}

static void append_prod_defaults(DefaultsCtx& c)
{
    // only the dim overload (prod(x, dim)) takes keepdim; the full-reduction
    // overload prod(x) has no dim/keepdim inputs, and appending keepdim here
    // would yield [input, keepdim, dtype], which no level-2 pattern matches
    if (c.has_input_name("dim"))
    {
        if (!c.has_input_name("keepdim"))
            c.add_const("keepdim", false);
        if (!c.has_input_name("dtype"))
            c.add_const("dtype", Parameter());
        c.reorder_inputs({"self", "dim", "keepdim", "dtype"});
    }
    else
    {
        // full-reduction prod(x): self (+ optional dtype) is already in order
        if (!c.has_input_name("dtype"))
            c.add_const("dtype", Parameter());
    }
}

static void append_cumsum_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("dtype"))
        c.add_const("dtype", Parameter());
}

static void append_cumprod_defaults(DefaultsCtx& c)
{
    // cumprod(x, dim) carries an omitted kwonly dtype default; append it so
    // the [input dim dtype] level-2 pattern matches
    if (!c.has_input_name("dtype"))
        c.add_const("dtype", Parameter());
    c.reorder_inputs({"self", "dim", "dtype"});
}

static void append_roll_defaults(DefaultsCtx& c)
{
    // roll(x, shifts) with omitted dims=None: restore [input shifts dims]
    if (!c.has_input_name("dims"))
        c.add_const("dims", Parameter());
    c.reorder_inputs({"self", "shifts", "dims"});
}

static void append_repeat_interleave_defaults(DefaultsCtx& c)
{
    // repeat_interleave(x, repeats) omits dim/output_size defaults
    if (!c.has_input_name("dim"))
        c.add_const("dim", Parameter());
    if (!c.has_input_name("output_size"))
        c.add_const("output_size", Parameter());
    c.reorder_inputs({"self", "repeats", "dim", "output_size"});
}

static void append_topk_defaults(DefaultsCtx& c)
{
    // topk(x, k) omits dim=-1/largest=True/sorted=True defaults; torch's
    // sorted default is True so the emitted values are in descending order
    if (!c.has_input_name("dim"))
        c.add_const("dim", -1);
    if (!c.has_input_name("largest"))
        c.add_const("largest", true);
    if (!c.has_input_name("sorted"))
        c.add_const("sorted", true);
    c.reorder_inputs({"self", "k", "dim", "largest", "sorted"});
}

static void append_istft_defaults(DefaultsCtx& c)
{
    // dynamo omits istft trailing defaults (onesided/length/return_complex)
    if (!c.has_input_name("onesided"))
        c.add_const("onesided", Parameter());
    if (!c.has_input_name("length"))
        c.add_const("length", Parameter());
    if (!c.has_input_name("return_complex"))
        c.add_const("return_complex", false);
}

static void append_cross_defaults(DefaultsCtx& c)
{
    // cross(x, y) omits the dim=None default; append the null dim slot so
    // the [input other dim] level-2 pattern matches
    if (!c.has_input_name("dim"))
        c.add_const("dim", Parameter());
}

static void append_index_put_defaults(DefaultsCtx& c)
{
    // index_put(x, indices, values) omits accumulate=False; append it so
    // the [input indices values accumulate] level-2 pattern matches
    if (!c.has_input_name("accumulate"))
        c.add_const("accumulate", false);
    c.reorder_inputs({"self", "indices", "values", "accumulate"});
}

static void append_cat_stack_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("dim"))
        c.add_const("dim", 0);
}

static void append_chunk_unbind_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("dim"))
        c.add_const("dim", 0);
}

static void append_split_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("dim"))
        c.add_const("dim", 0);
}

static void append_diag_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("diagonal"))
        c.add_const("diagonal", 0);
}

static void append_clone_defaults(DefaultsCtx& c)
{
    // dynamo omits memory_format; torch.clone defaults to preserve_format(=1)
    if (!c.has_input_name("memory_format"))
        c.add_const("memory_format", 1);
}

static void append_addmm_defaults(DefaultsCtx& c)
{
    // beta/alpha are trailing keyword-only defaults; a call supplying only
    // one of them (e.g. addmm(b, m1, m2, alpha=2)) serializes the omitted
    // one out of place, so restore the canonical [self mat1 mat2 beta alpha]
    // order for the level-2 pattern
    if (!c.has_input_name("beta"))
        c.add_const("beta", 1);
    if (!c.has_input_name("alpha"))
        c.add_const("alpha", 1);
    c.reorder_inputs({"self", "mat1", "mat2", "beta", "alpha"});
}

static void append_baddbmm_defaults(DefaultsCtx& c)
{
    // baddbmm(self, batch1, batch2) omits beta=1/alpha=1; restore the
    // canonical [self batch1 batch2 beta alpha] order for the level-2 pattern
    if (!c.has_input_name("beta"))
        c.add_const("beta", 1);
    if (!c.has_input_name("alpha"))
        c.add_const("alpha", 1);
    c.reorder_inputs({"self", "batch1", "batch2", "beta", "alpha"});
}

static void append_linalg_vector_norm_defaults(DefaultsCtx& c)
{
    // dtype is a keyword-only trailing default; restore the canonical
    // [self ord dim keepdim dtype] order when it was serialized ahead of
    // the omitted ord/dim/keepdim defaults
    if (!c.has_input_name("ord"))
        c.add_const("ord", 2.0f);
    if (!c.has_input_name("dim"))
        c.add_const("dim", Parameter());
    if (!c.has_input_name("keepdim"))
        c.add_const("keepdim", false);
    if (!c.has_input_name("dtype"))
        c.add_const("dtype", Parameter());
    c.reorder_inputs({"self", "ord", "dim", "keepdim", "dtype"});
}

static void append_weight_norm_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("dim"))
        c.add_const("dim", 0);
}

static void append_clamp_defaults(DefaultsCtx& c)
{
    if (!c.has_input_name("min"))
        c.add_const("min", Parameter());
    if (!c.has_input_name("max"))
        c.add_const("max", Parameter());
}

static void append_arange_defaults(DefaultsCtx& c)
{
    // dynamo omits the dtype default (None) from aten::arange, leaving
    // [end device pin_memory] (or the start/start_step variants) which
    // matches no torch_arange level-2 pattern and survives to codegen as an
    // invalid "aten::arange(...)" python line; restore the canonical order
    // the pt2 torch_arange_5/6/7 patterns expect
    if (!c.has_input_name("dtype"))
        c.add_const("dtype", Parameter());
    if (c.type == "aten::arange.start_step")
        c.reorder_inputs({"start", "end", "step", "dtype", "device", "pin_memory"});
    else if (c.type == "aten::arange.start")
        c.reorder_inputs({"start", "end", "dtype", "device", "pin_memory"});
    else
        c.reorder_inputs({"end", "dtype", "device", "pin_memory"});
}

static void append_zeros_ones_defaults(DefaultsCtx& c)
{
    // dynamo omits the dtype default (None) from aten::zeros/ones, leaving
    // [size device pin_memory] which matches neither the level-2 fold nor
    // the torch.zeros rewrite; restore the canonical [size dtype device
    // pin_memory] order so the constant folds to an Attribute
    if (!c.has_input_name("dtype"))
        c.add_const("dtype", Parameter());
    c.reorder_inputs({"self", "size", "dtype", "layout", "device", "pin_memory"});
}

static void append_full_defaults(DefaultsCtx& c)
{
    // same dtype-default omission for aten::full; restore
    // [size fill_value dtype device pin_memory] for the level-2 fold
    if (!c.has_input_name("dtype"))
        c.add_const("dtype", Parameter());
    c.reorder_inputs({"self", "size", "fill_value", "dtype", "layout", "device", "pin_memory"});
}

static void append_new_full_defaults(DefaultsCtx& c)
{
    // Tensor.new_full(self, size, fill_value) omits the dtype/layout/
    // device defaults; restore the canonical order for the level-2 fold
    if (!c.has_input_name("dtype"))
        c.add_const("dtype", Parameter());
    if (!c.has_input_name("layout"))
        c.add_const("layout", Parameter());
    if (!c.has_input_name("device"))
        c.add_const("device", Parameter());
    c.reorder_inputs({"self", "size", "fill_value", "dtype", "layout", "device", "pin_memory"});
}

static void append_new_zeros_defaults(DefaultsCtx& c)
{
    // dynamo emits Tensor.new_zeros(self, size, pin_memory); the pnnx
    // pass_level2 pattern expects input size dtype layout device pin_memory.
    // GraphRewriter matches constant inputs POSITIONALLY, so the final
    // order must be exactly: input size dtype layout device pin_memory.
    std::vector<Operand*> old_inputs = c.op->inputs;
    std::vector<std::string> old_names = c.op->inputnames;

    // detach op from all old inputs; keep the interesting ones
    for (size_t j = 0; j < old_inputs.size(); j++)
    {
        auto& cons = old_inputs[j]->consumers;
        cons.erase(std::find(cons.begin(), cons.end(), c.op));
    }

    c.op->inputs.clear();
    c.op->inputnames.clear();

    Operand* self_op = 0;
    Operand* size_op = 0;
    Operand* dtype_op = 0;
    Operand* layout_op = 0;
    Operand* device_op = 0;
    bool have_pin = false;
    for (size_t j = 0; j < old_names.size(); j++)
    {
        std::string n = old_names[j];
        if (n == "self" || n == "input")
            self_op = old_inputs[j];
        else if (n == "size")
            size_op = old_inputs[j];
        else if (n == "dtype")
            dtype_op = old_inputs[j];
        else if (n == "layout")
            layout_op = old_inputs[j];
        else if (n == "device")
            device_op = old_inputs[j];
        else if (n == "pin_memory")
            have_pin = true;
    }

    // rebuild in pattern order: input size dtype layout device pin_memory
    if (self_op)
    {
        c.op->inputs.push_back(self_op);
        c.op->inputnames.push_back("input");
        self_op->consumers.push_back(c.op);
    }
    if (size_op)
    {
        c.op->inputs.push_back(size_op);
        c.op->inputnames.push_back("size");
        size_op->consumers.push_back(c.op);
    }
    // (size is always present for new_*; if it were missing the pattern
    //  simply will not match and the op stays as-is, which is safe)
    if (dtype_op)
    {
        // keep an explicit dtype constant when dynamo emitted one
        // (e.g. x.new_empty(..., dtype=torch.long)); otherwise null means
        // "inherit self's dtype", matching the pattern default
        c.op->inputs.push_back(dtype_op);
        c.op->inputnames.push_back("dtype");
        dtype_op->consumers.push_back(c.op);
    }
    else
    {
        c.add_const("dtype", Parameter());
    }
    if (layout_op)
    {
        c.op->inputs.push_back(layout_op);
        c.op->inputnames.push_back("layout");
        layout_op->consumers.push_back(c.op);
    }
    else
    {
        c.add_const("layout", Parameter());
    }
    if (device_op)
    {
        c.op->inputs.push_back(device_op);
        c.op->inputnames.push_back("device");
        device_op->consumers.push_back(c.op);
    }
    else
    {
        c.add_const("device", Parameter());
    }
    c.add_const("pin_memory", have_pin); // fresh constant either way
}

static void append_ones_like_defaults(DefaultsCtx& c)
{
    // dynamo emits input [dtype|fill_value] pin_memory; the pnnx pattern
    // expects input dtype layout device requires_grad memory_format
    // (full_like additionally carries fill_value)
    std::vector<Operand*> old_inputs = c.op->inputs;
    std::vector<std::string> old_names = c.op->inputnames;
    for (size_t j = 0; j < old_names.size(); j++)
    {
        if (old_names[j] != "self" && old_names[j] != "input" && old_names[j] != "dtype")
        {
            // drop irrelevant inputs like pin_memory and clean up consumer refs
            auto& cons = old_inputs[j]->consumers;
            cons.erase(std::find(cons.begin(), cons.end(), c.op));
        }
    }

    c.op->inputs.clear();
    c.op->inputnames.clear();

    int found_input = -1;
    for (size_t j = 0; j < old_names.size(); j++)
        if (old_names[j] == "self" || old_names[j] == "input")
        {
            found_input = (int)j;
            break;
        }
    if (found_input != -1)
    {
        c.op->inputs.push_back(old_inputs[found_input]);
        c.op->inputnames.push_back("input");
    }

    int found_dtype = -1;
    for (size_t j = 0; j < old_names.size(); j++)
        if (old_names[j] == "dtype")
        {
            found_dtype = (int)j;
            break;
        }
    if (found_dtype != -1)
    {
        c.op->inputs.push_back(old_inputs[found_dtype]);
        c.op->inputnames.push_back("dtype");
    }
    else
    {
        c.add_const("dtype", Parameter());
    }

    // full_like: dynamo passes fill_value as a scalar input; the pnnx
    // pattern wants it as the first input (input fill_value dtype ...)
    if (c.type == "aten::full_like")
    {
        int found_fill = -1;
        for (size_t j = 0; j < old_names.size(); j++)
            if (old_names[j] == "fill_value")
            {
                found_fill = (int)j;
                break;
            }
        if (found_fill != -1)
        {
            Operand* fill = old_inputs[found_fill];
            c.op->inputs.insert(c.op->inputs.begin() + 1, fill);
            c.op->inputnames.insert(c.op->inputnames.begin() + 1, "fill_value");
            // the general cleanup above erased fill_value from its consumer
            // list; restore the reverse edge so the level-2 torch_full_like
            // pattern (which matches constant outputs by consumer count) can
            // fire instead of leaving the raw aten::full_like node behind
            fill->consumers.push_back(c.op);
        }
        else
        {
            // fill_value missing: synthesize a constant 0 and place it at
            // index 1 (new_constant appends at the end)
            c.add_const("fill_value", 0);
            Operand* fill = c.op->inputs.back();
            c.op->inputs.pop_back();
            c.op->inputs.insert(c.op->inputs.begin() + 1, fill);
            c.op->inputnames.pop_back();
            c.op->inputnames.insert(c.op->inputnames.begin() + 1, "fill_value");
        }
    }

    c.add_const("layout", Parameter());
    c.add_const("device", Parameter());
    c.add_const("requires_grad", false);
    c.add_const("memory_format", Parameter());
}

// type -> handler dispatch table. the first matching entry runs; every type
// listed here maps to exactly one handler so the order only matters for
// readability. adding support for a new aten op = one handler + one entry.
static const struct DefaultsDispatchEntry
{
    const char* type;
    DefaultsHandler handler;
} defaults_dispatch[] = {
    {"aten::conv1d", &append_conv_defaults},
    {"aten::conv2d", &append_conv_defaults},
    {"aten::conv3d", &append_conv_defaults},
    {"aten::batch_norm", &append_batch_norm_defaults},
    {"aten::add", &append_add_defaults},
    {"aten::max_pool1d", &append_max_pool_defaults},
    {"aten::max_pool2d", &append_max_pool_defaults},
    {"aten::max_pool3d", &append_max_pool_defaults},
    {"aten::max_pool1d_with_indices", &append_max_pool_defaults},
    {"aten::max_pool2d_with_indices", &append_max_pool_defaults},
    {"aten::max_pool3d_with_indices", &append_max_pool_defaults},
    {"aten::avg_pool1d", &append_avg_pool_defaults},
    {"aten::avg_pool2d", &append_avg_pool_defaults},
    {"aten::avg_pool3d", &append_avg_pool_defaults},
    {"aten::argmax", &append_argmax_defaults},
    {"aten::argmin", &append_argmax_defaults},
    {"aten::sum", &append_sum_mean_defaults},
    {"aten::mean", &append_sum_mean_defaults},
    {"aten::var", &append_var_std_defaults},
    {"aten::std", &append_var_std_defaults},
    {"aten::softmax", &append_softmax_defaults},
    {"aten::log_softmax", &append_softmax_defaults},
    {"aten::pad", &append_pad_defaults},
    {"aten::to", &append_to_defaults},
    {"aten::contiguous", &append_contiguous_defaults},
    {"aten::slice", &append_slice_defaults},
    {"aten::slice_scatter", &append_slice_scatter_defaults},
    {"aten::flatten", &append_flatten_defaults},
    {"aten::celu", &append_celu_defaults},
    {"aten::elu", &append_elu_defaults},
    {"aten::hardshrink", &append_hardshrink_defaults},
    {"aten::hardtanh", &append_hardtanh_defaults},
    {"aten::leaky_relu", &append_leaky_relu_defaults},
    {"aten::softplus", &append_softplus_defaults},
    {"aten::softshrink", &append_softshrink_defaults},
    {"aten::rrelu", &append_rrelu_defaults},
    {"aten::pairwise_distance", &append_pairwise_distance_defaults},
    {"aten::linear", &append_linear_defaults},
    {"aten::as_strided", &append_as_strided_defaults},
    {"aten::tril", &append_tril_defaults},
    {"aten::rms_norm", &append_rms_norm_defaults},
    {"aten::scaled_dot_product_attention", &append_scaled_dot_product_attention_defaults},
    {"aten::embedding", &append_embedding_defaults},
    {"aten::glu", &append_glu_defaults},
    {"aten::conv_transpose1d", &append_conv_transpose_defaults},
    {"aten::conv_transpose2d", &append_conv_transpose_defaults},
    {"aten::conv_transpose3d", &append_conv_transpose_defaults},
    {"aten::amax", &append_amax_defaults},
    {"aten::amin", &append_amax_defaults},
    {"aten::max", &append_max_min_defaults},
    {"aten::min", &append_max_min_defaults},
    {"aten::logsumexp", &append_logsumexp_defaults},
    {"aten::prod", &append_prod_defaults},
    {"aten::cumsum", &append_cumsum_defaults},
    {"aten::cumprod", &append_cumprod_defaults},
    {"aten::roll", &append_roll_defaults},
    {"aten::repeat_interleave", &append_repeat_interleave_defaults},
    {"aten::topk", &append_topk_defaults},
    {"aten::istft", &append_istft_defaults},
    {"aten::cross", &append_cross_defaults},
    {"aten::index_put", &append_index_put_defaults},
    {"aten::index_put_", &append_index_put_defaults},
    {"aten::cat", &append_cat_stack_defaults},
    {"aten::stack", &append_cat_stack_defaults},
    {"aten::chunk", &append_chunk_unbind_defaults},
    {"aten::unbind", &append_chunk_unbind_defaults},
    {"aten::split", &append_split_defaults},
    {"aten::split_with_sizes", &append_split_defaults},
    {"aten::tensor_split", &append_split_defaults},
    {"aten::diag", &append_diag_defaults},
    {"aten::clone", &append_clone_defaults},
    {"aten::addmm", &append_addmm_defaults},
    {"aten::baddbmm", &append_baddbmm_defaults},
    {"aten::linalg_vector_norm", &append_linalg_vector_norm_defaults},
    {"aten::_weight_norm", &append_weight_norm_defaults},
    {"aten::clamp", &append_clamp_defaults},
    {"aten::arange", &append_arange_defaults},
    {"aten::arange.start", &append_arange_defaults},
    {"aten::arange.start_step", &append_arange_defaults},
    {"aten::zeros", &append_zeros_ones_defaults},
    {"aten::ones", &append_zeros_ones_defaults},
    {"aten::full", &append_full_defaults},
    {"aten::new_full", &append_new_full_defaults},
    {"aten::new_zeros", &append_new_zeros_defaults},
    {"aten::new_ones", &append_new_zeros_defaults},
    {"aten::new_empty", &append_new_zeros_defaults},
    {"aten::ones_like", &append_ones_like_defaults},
    {"aten::zeros_like", &append_ones_like_defaults},
    {"aten::rand_like", &append_ones_like_defaults},
    {"aten::randn_like", &append_ones_like_defaults},
    {"aten::empty_like", &append_ones_like_defaults},
    {"aten::full_like", &append_ones_like_defaults},
};

// append default scalar inputs for aten operators that omitted default kwargs
// dynamo omits schema default arguments; fill them by parameter name here,
// keeping the input order consistent with the pass_level2 patterns (omitted
// ones are trailing defaults, so appending keeps the order)
void append_default_kwargs(Graph& g, Operator* op, const std::string& type, const std::vector<std::string>& inputnames, int& constant_index)
{
    DefaultsCtx ctx = {g, op, type, inputnames, constant_index};

    for (size_t i = 0; i < sizeof(defaults_dispatch) / sizeof(defaults_dispatch[0]); i++)
    {
        if (strcmp(defaults_dispatch[i].type, type.c_str()) == 0)
        {
            defaults_dispatch[i].handler(ctx);
            return;
        }
    }
}

} // namespace pnnx
