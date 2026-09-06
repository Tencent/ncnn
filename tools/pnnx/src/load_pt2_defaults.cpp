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
static bool has_input_name(const std::vector<std::string>& inputnames, const std::string& name)
{
    for (size_t i = 0; i < inputnames.size(); i++)
        if (inputnames[i] == name)
            return true;
    return false;
}

// find the prim::Constant value of an input by parameter name
// used for defaults that depend on other params, e.g. stride = kernel_size
static Parameter find_input_value(Operator* op, const std::vector<std::string>& inputnames, const std::string& name)
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

void append_default_kwargs(Graph& g, Operator* op, const std::string& type, const std::vector<std::string>& inputnames, int& constant_index)
{
    // add a default constant input and keep op->inputnames in sync with op->inputs
    auto add_const = [&](const std::string& name, const Parameter& value) {
        new_constant(g, op, value, constant_index);
        op->inputnames.push_back(name);
    };

    // reorder op inputs/inputnames to a canonical schema order (the exported
    // overload may omit middle defaults, and blindly appending them would
    // misalign the level-2 patterns that match by position)
    auto reorder_inputs = [&](const std::vector<std::string>& order) {
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
    };

    if (type == "aten::conv1d" || type == "aten::conv2d" || type == "aten::conv3d")
    {
        int dim = 2;
        if (type == "aten::conv1d")
            dim = 1;
        else if (type == "aten::conv3d")
            dim = 3;

        std::vector<int> ones(dim, 1);
        std::vector<int> zeros(dim, 0);

        if (!has_input_name(inputnames, "stride"))
            add_const("stride", ones);
        if (!has_input_name(inputnames, "padding"))
            add_const("padding", zeros);
        if (!has_input_name(inputnames, "dilation"))
            add_const("dilation", ones);
        if (!has_input_name(inputnames, "groups"))
            add_const("groups", 1);
    }
    else if (type == "aten::batch_norm")
    {
        if (!has_input_name(inputnames, "training"))
            add_const("training", false);
        if (!has_input_name(inputnames, "momentum"))
            add_const("momentum", 0.1f);
        if (!has_input_name(inputnames, "eps"))
            add_const("eps", 1e-5f);
        if (!has_input_name(inputnames, "cudnn_enabled"))
            add_const("cudnn_enabled", true);
    }
    else if (type == "aten::add")
    {
        if (!has_input_name(inputnames, "alpha"))
            add_const("alpha", 1);
    }
    else if (type == "aten::max_pool1d" || type == "aten::max_pool2d" || type == "aten::max_pool3d"
             || type == "aten::max_pool1d_with_indices" || type == "aten::max_pool2d_with_indices" || type == "aten::max_pool3d_with_indices")
    {
        int dim = 2;
        if (type == "aten::max_pool1d" || type == "aten::max_pool1d_with_indices")
            dim = 1;
        else if (type == "aten::max_pool3d" || type == "aten::max_pool3d_with_indices")
            dim = 3;

        // torch max_pool stride defaults to kernel_size
        Parameter kernel = find_input_value(op, inputnames, "kernel_size");

        if (!has_input_name(inputnames, "stride"))
        {
            if (kernel.type == 5)
                add_const("stride", kernel.ai);
            else
                add_const("stride", std::vector<int>(dim, 1));
        }
        if (!has_input_name(inputnames, "padding"))
            add_const("padding", std::vector<int>(dim, 0));
        if (!has_input_name(inputnames, "dilation"))
            add_const("dilation", std::vector<int>(dim, 1));
        if (!has_input_name(inputnames, "ceil_mode"))
            add_const("ceil_mode", false);
    }
    else if (type == "aten::avg_pool1d" || type == "aten::avg_pool2d" || type == "aten::avg_pool3d")
    {
        int dim = 2;
        if (type == "aten::avg_pool1d")
            dim = 1;
        else if (type == "aten::avg_pool3d")
            dim = 3;

        // torch avg_pool stride defaults to kernel_size
        Parameter kernel = find_input_value(op, inputnames, "kernel_size");

        if (!has_input_name(inputnames, "stride"))
        {
            if (kernel.type == 5)
                add_const("stride", kernel.ai);
            else
                add_const("stride", std::vector<int>(dim, 1));
        }
        if (!has_input_name(inputnames, "padding"))
            add_const("padding", std::vector<int>(dim, 0));
        if (!has_input_name(inputnames, "ceil_mode"))
            add_const("ceil_mode", false);
        if (!has_input_name(inputnames, "count_include_pad"))
            add_const("count_include_pad", true);
        if (type != "aten::avg_pool1d")
        {
            if (!has_input_name(inputnames, "divisor_override"))
                add_const("divisor_override", Parameter());
        }
    }
    else if (type == "aten::argmax" || type == "aten::argmin")
    {
        // dim=None means full reduction (torch.argmax(x) without dim)
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", Parameter());
        if (!has_input_name(inputnames, "keepdim"))
            add_const("keepdim", false);
    }
    else if (type == "aten::sum" || type == "aten::mean")
    {
        // mean.default / sum full-reduction versions have no keepdim argument
        if (has_input_name(inputnames, "dim"))
        {
            if (!has_input_name(inputnames, "keepdim"))
                add_const("keepdim", false);
            if (!has_input_name(inputnames, "dtype"))
                add_const("dtype", Parameter());
            // keepdim sits before dtype in the schema; when dtype was already
            // serialized but keepdim omitted, reorder so the level-2 pattern
            // does not read dtype as keepdim
            reorder_inputs({"self", "dim", "keepdim", "dtype"});
        }
        else
        {
            // full-reduction overload: self (+ optional dtype) is already in order
            if (!has_input_name(inputnames, "dtype"))
                add_const("dtype", Parameter());
        }
    }
    else if (type == "aten::var" || type == "aten::std")
    {
        // aten::var/std.dim overloads serialize an unbiased argument; the
        // .correction overload serializes correction. never mix the two families
        // (a 5-input node matches no level-2 pattern), and keep the defaults in
        // canonical schema order so the torch_std/torch_var rewrites match.
        if (has_input_name(inputnames, "unbiased"))
        {
            if (!has_input_name(inputnames, "keepdim"))
                add_const("keepdim", false);
            reorder_inputs({"self", "dim", "unbiased", "keepdim"});
        }
        else if (has_input_name(inputnames, "correction"))
        {
            if (!has_input_name(inputnames, "keepdim"))
                add_const("keepdim", false);
            reorder_inputs({"self", "dim", "correction", "keepdim"});
        }
        else if (has_input_name(inputnames, "dim"))
        {
            // dim overload with the default unbiased/keepdim omitted
            if (!has_input_name(inputnames, "unbiased"))
                add_const("unbiased", true);
            if (!has_input_name(inputnames, "keepdim"))
                add_const("keepdim", false);
            reorder_inputs({"self", "dim", "unbiased", "keepdim"});
        }
        else
        {
            // reduce-all overload (self only, serialized under .correction)
            if (!has_input_name(inputnames, "correction"))
                add_const("correction", 1);
            if (!has_input_name(inputnames, "keepdim"))
                add_const("keepdim", false);
        }
    }
    else if (type == "aten::softmax" || type == "aten::log_softmax")
    {
        if (!has_input_name(inputnames, "dtype"))
            add_const("dtype", Parameter());
    }
    else if (type == "aten::pad")
    {
        if (!has_input_name(inputnames, "mode"))
            add_const("mode", std::string("constant"));
        if (!has_input_name(inputnames, "value"))
            add_const("value", Parameter());
    }
    else if (type == "aten::to")
    {
        // dynamo emits to.dtype / to.device / to.dtype_layout overloads with
        // the trailing defaults omitted; fill them to match the Tensor_to
        // patterns. the dtype_layout overload also carries pin_memory between
        // device and non_blocking.
        if (!has_input_name(inputnames, "non_blocking"))
            add_const("non_blocking", false);
        if (!has_input_name(inputnames, "copy"))
            add_const("copy", false);
        if (!has_input_name(inputnames, "memory_format"))
            add_const("memory_format", Parameter());
        if (has_input_name(inputnames, "layout"))
        {
            if (!has_input_name(inputnames, "pin_memory"))
                add_const("pin_memory", false);
            reorder_inputs({"self", "dtype", "layout", "device", "pin_memory", "non_blocking", "copy", "memory_format"});
        }
        else if (has_input_name(inputnames, "device"))
        {
            reorder_inputs({"self", "device", "dtype", "non_blocking", "copy", "memory_format"});
        }
        else
        {
            reorder_inputs({"self", "dtype", "non_blocking", "copy", "memory_format"});
        }
    }
    else if (type == "aten::contiguous")
    {
        // eliminate_contiguous expects 2 inputs (input + memory_format); dynamo omits memory_format
        if (!has_input_name(inputnames, "memory_format"))
            add_const("memory_format", Parameter());
    }
    else if (type == "aten::slice")
    {
        if (!has_input_name(inputnames, "step"))
            add_const("step", 1);
    }
    else if (type == "aten::slice_scatter")
    {
        // slice_scatter(self, src) omits the dim=0/start=None defaults; the
        // [input src dim start end step] level-2 pattern needs every slot or the
        // node survives as an illegal aten::slice_scatter call
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", 0);
        if (!has_input_name(inputnames, "start"))
            add_const("start", Parameter());
        if (!has_input_name(inputnames, "end"))
            add_const("end", INT_MAX);
        if (!has_input_name(inputnames, "step"))
            add_const("step", 1);
        reorder_inputs({"self", "src", "dim", "start", "end", "step"});
    }
    else if (type == "aten::flatten")
    {
        if (!has_input_name(inputnames, "start_dim"))
            add_const("start_dim", 0);
        if (!has_input_name(inputnames, "end_dim"))
            add_const("end_dim", -1);
    }
    else if (type == "aten::celu")
    {
        if (!has_input_name(inputnames, "alpha"))
            add_const("alpha", 1.0f);
    }
    else if (type == "aten::elu")
    {
        if (!has_input_name(inputnames, "alpha"))
            add_const("alpha", 1.0f);
        if (!has_input_name(inputnames, "scale"))
            add_const("scale", 1.0f);
        if (!has_input_name(inputnames, "input_scale"))
            add_const("input_scale", 1.0f);
    }
    else if (type == "aten::hardshrink")
    {
        if (!has_input_name(inputnames, "lambd"))
            add_const("lambd", 0.5f);
    }
    else if (type == "aten::hardtanh")
    {
        if (!has_input_name(inputnames, "min_val"))
            add_const("min_val", -1.0f);
        if (!has_input_name(inputnames, "max_val"))
            add_const("max_val", 1.0f);
    }
    else if (type == "aten::leaky_relu")
    {
        if (!has_input_name(inputnames, "negative_slope"))
            add_const("negative_slope", 0.01f);
    }
    else if (type == "aten::softplus")
    {
        if (!has_input_name(inputnames, "beta"))
            add_const("beta", 1.0f);
        if (!has_input_name(inputnames, "threshold"))
            add_const("threshold", 20.0f);
    }
    else if (type == "aten::softshrink")
    {
        if (!has_input_name(inputnames, "lambd"))
            add_const("lambd", 0.5f);
    }
    else if (type == "aten::rrelu")
    {
        if (!has_input_name(inputnames, "lower"))
            add_const("lower", 0.125f);
        if (!has_input_name(inputnames, "upper"))
            add_const("upper", 1.0f / 3.0f);
        if (!has_input_name(inputnames, "training"))
            add_const("training", false);
        if (!has_input_name(inputnames, "generator"))
            add_const("generator", Parameter());
    }
    else if (type == "aten::pairwise_distance")
    {
        if (!has_input_name(inputnames, "p"))
            add_const("p", 2);
        if (!has_input_name(inputnames, "eps"))
            add_const("eps", 1e-6f);
        if (!has_input_name(inputnames, "keepdim"))
            add_const("keepdim", false);
    }
    else if (type == "aten::linear")
    {
        if (!has_input_name(inputnames, "bias"))
            add_const("bias", Parameter());
    }
    else if (type == "aten::as_strided")
    {
        // as_strided(self, size, stride) omits storage_offset=0; append it so
        // the [input size stride storage_offset] level-2 pattern matches
        if (!has_input_name(inputnames, "storage_offset"))
            add_const("storage_offset", 0);
    }
    else if (type == "aten::tril")
    {
        // tril(input) omits the diagonal=0 default; append it so the
        // [input diagonal] level-2 pattern matches
        if (!has_input_name(inputnames, "diagonal"))
            add_const("diagonal", 0);
    }
    else if (type == "aten::rms_norm")
    {
        if (!has_input_name(inputnames, "weight"))
            add_const("weight", Parameter());
        if (!has_input_name(inputnames, "eps"))
            add_const("eps", Parameter());
    }
    else if (type == "aten::scaled_dot_product_attention")
    {
        // dynamo omits intermediate default args and enable_gqa moves up to the
        // attn_mask slot; reorder to match the pattern:
        // query key value attn_mask dropout_p is_causal scale enable_gqa
        static const std::vector<std::string> order = {"query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa"};

        std::vector<Operand*> old_inputs = op->inputs;
        std::vector<std::string> old_names = op->inputnames;
        op->inputs.clear();
        op->inputnames.clear();

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
                op->inputs.push_back(old_inputs[found]);
                op->inputnames.push_back(name);
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

                new_constant(g, op, v, constant_index);
                op->inputnames.push_back(name);
            }
        }
    }
    else if (type == "aten::embedding")
    {
        if (!has_input_name(inputnames, "padding_idx"))
            add_const("padding_idx", -1);
        if (!has_input_name(inputnames, "scale_grad_by_freq"))
            add_const("scale_grad_by_freq", false);
        if (!has_input_name(inputnames, "sparse"))
            add_const("sparse", false);
    }
    else if (type == "aten::glu")
    {
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", -1);
    }
    else if (type == "aten::conv_transpose1d" || type == "aten::conv_transpose2d" || type == "aten::conv_transpose3d")
    {
        int dim = 2;
        if (type == "aten::conv_transpose1d")
            dim = 1;
        else if (type == "aten::conv_transpose3d")
            dim = 3;

        if (!has_input_name(inputnames, "stride"))
            add_const("stride", std::vector<int>(dim, 1));
        if (!has_input_name(inputnames, "padding"))
            add_const("padding", std::vector<int>(dim, 0));
        if (!has_input_name(inputnames, "output_padding"))
            add_const("output_padding", std::vector<int>(dim, 0));
        if (!has_input_name(inputnames, "groups"))
            add_const("groups", 1);
        if (!has_input_name(inputnames, "dilation"))
            add_const("dilation", std::vector<int>(dim, 1));
    }
    else if (type == "aten::amax" || type == "aten::amin")
    {
        // dim defaults to None (reduce all); when omitted add the null dim slot
        // so [self, dim, keepdim] matches the level-2 pattern instead of
        // leaving a keepdim-only node that no pattern rewrites
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", Parameter());
        if (!has_input_name(inputnames, "keepdim"))
            add_const("keepdim", false);
        reorder_inputs({"self", "dim", "keepdim"});
    }
    else if (type == "aten::max" || type == "aten::min")
    {
        // max.dim/min.dim omit keepdim when it is False; inputnames has dim but
        // lacks keepdim (max.other/default have no dim and are unaffected)
        if (has_input_name(inputnames, "dim") && !has_input_name(inputnames, "keepdim"))
            add_const("keepdim", false);
    }
    else if (type == "aten::logsumexp")
    {
        if (!has_input_name(inputnames, "keepdim"))
            add_const("keepdim", false);
    }
    else if (type == "aten::prod")
    {
        // only the dim overload (prod(x, dim)) takes keepdim; the full-reduction
        // overload prod(x) has no dim/keepdim inputs, and appending keepdim here
        // would yield [input, keepdim, dtype], which no level-2 pattern matches
        if (has_input_name(inputnames, "dim"))
        {
            if (!has_input_name(inputnames, "keepdim"))
                add_const("keepdim", false);
            if (!has_input_name(inputnames, "dtype"))
                add_const("dtype", Parameter());
            reorder_inputs({"self", "dim", "keepdim", "dtype"});
        }
        else
        {
            // full-reduction prod(x): self (+ optional dtype) is already in order
            if (!has_input_name(inputnames, "dtype"))
                add_const("dtype", Parameter());
        }
    }
    else if (type == "aten::cumsum")
    {
        if (!has_input_name(inputnames, "dtype"))
            add_const("dtype", Parameter());
    }
    else if (type == "aten::cumprod")
    {
        // cumprod(x, dim) carries an omitted kwonly dtype default; append it so
        // the [input dim dtype] level-2 pattern matches
        if (!has_input_name(inputnames, "dtype"))
            add_const("dtype", Parameter());
        reorder_inputs({"self", "dim", "dtype"});
    }
    else if (type == "aten::roll")
    {
        // roll(x, shifts) with omitted dims=None: restore [input shifts dims]
        if (!has_input_name(inputnames, "dims"))
            add_const("dims", Parameter());
        reorder_inputs({"self", "shifts", "dims"});
    }
    else if (type == "aten::repeat_interleave")
    {
        // repeat_interleave(x, repeats) omits dim/output_size defaults
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", Parameter());
        if (!has_input_name(inputnames, "output_size"))
            add_const("output_size", Parameter());
        reorder_inputs({"self", "repeats", "dim", "output_size"});
    }
    else if (type == "aten::topk")
    {
        // topk(x, k) omits dim=-1/largest=True/sorted=True defaults; torch's
        // sorted default is True so the emitted values are in descending order
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", -1);
        if (!has_input_name(inputnames, "largest"))
            add_const("largest", true);
        if (!has_input_name(inputnames, "sorted"))
            add_const("sorted", true);
        reorder_inputs({"self", "k", "dim", "largest", "sorted"});
    }
    else if (type == "aten::istft")
    {
        // dynamo omits istft trailing defaults (onesided/length/return_complex)
        if (!has_input_name(inputnames, "onesided"))
            add_const("onesided", Parameter());
        if (!has_input_name(inputnames, "length"))
            add_const("length", Parameter());
        if (!has_input_name(inputnames, "return_complex"))
            add_const("return_complex", false);
    }
    else if (type == "aten::cross")
    {
        // cross(x, y) omits the dim=None default; append the null dim slot so
        // the [input other dim] level-2 pattern matches
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", Parameter());
    }
    else if (type == "aten::index_put" || type == "aten::index_put_")
    {
        // index_put(x, indices, values) omits accumulate=False; append it so
        // the [input indices values accumulate] level-2 pattern matches
        if (!has_input_name(inputnames, "accumulate"))
            add_const("accumulate", false);
        reorder_inputs({"self", "indices", "values", "accumulate"});
    }
    else if (type == "aten::cat" || type == "aten::stack")
    {
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", 0);
    }
    else if (type == "aten::chunk" || type == "aten::unbind")
    {
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", 0);
    }
    else if (type == "aten::split" || type == "aten::split_with_sizes" || type == "aten::tensor_split")
    {
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", 0);
    }
    else if (type == "aten::diag")
    {
        if (!has_input_name(inputnames, "diagonal"))
            add_const("diagonal", 0);
    }
    else if (type == "aten::clone")
    {
        // dynamo omits memory_format; torch.clone defaults to preserve_format(=1)
        if (!has_input_name(inputnames, "memory_format"))
            add_const("memory_format", 1);
    }
    else if (type == "aten::addmm")
    {
        // beta/alpha are trailing keyword-only defaults; a call supplying only
        // one of them (e.g. addmm(b, m1, m2, alpha=2)) serializes the omitted
        // one out of place, so restore the canonical [self mat1 mat2 beta alpha]
        // order for the level-2 pattern
        if (!has_input_name(inputnames, "beta"))
            add_const("beta", 1);
        if (!has_input_name(inputnames, "alpha"))
            add_const("alpha", 1);
        reorder_inputs({"self", "mat1", "mat2", "beta", "alpha"});
    }
    else if (type == "aten::baddbmm")
    {
        // baddbmm(self, batch1, batch2) omits beta=1/alpha=1; restore the
        // canonical [self batch1 batch2 beta alpha] order for the level-2 pattern
        if (!has_input_name(inputnames, "beta"))
            add_const("beta", 1);
        if (!has_input_name(inputnames, "alpha"))
            add_const("alpha", 1);
        reorder_inputs({"self", "batch1", "batch2", "beta", "alpha"});
    }
    else if (type == "aten::linalg_vector_norm")
    {
        // dtype is a keyword-only trailing default; restore the canonical
        // [self ord dim keepdim dtype] order when it was serialized ahead of
        // the omitted ord/dim/keepdim defaults
        if (!has_input_name(inputnames, "ord"))
            add_const("ord", 2.0f);
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", Parameter());
        if (!has_input_name(inputnames, "keepdim"))
            add_const("keepdim", false);
        if (!has_input_name(inputnames, "dtype"))
            add_const("dtype", Parameter());
        reorder_inputs({"self", "ord", "dim", "keepdim", "dtype"});
    }
    else if (type == "aten::_weight_norm")
    {
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", 0);
    }
    else if (type == "aten::clamp")
    {
        if (!has_input_name(inputnames, "min"))
            add_const("min", Parameter());
        if (!has_input_name(inputnames, "max"))
            add_const("max", Parameter());
    }
    else if (type == "aten::arange" || type == "aten::arange.start" || type == "aten::arange.start_step")
    {
        // dynamo omits the dtype default (None) from aten::arange, leaving
        // [end device pin_memory] (or the start/start_step variants) which
        // matches no torch_arange level-2 pattern and survives to codegen as an
        // invalid "aten::arange(...)" python line; restore the canonical order
        // the pt2 torch_arange_5/6/7 patterns expect
        if (!has_input_name(inputnames, "dtype"))
            add_const("dtype", Parameter());
        if (type == "aten::arange.start_step")
            reorder_inputs({"start", "end", "step", "dtype", "device", "pin_memory"});
        else if (type == "aten::arange.start")
            reorder_inputs({"start", "end", "dtype", "device", "pin_memory"});
        else
            reorder_inputs({"end", "dtype", "device", "pin_memory"});
    }
    else if (type == "aten::zeros" || type == "aten::ones")
    {
        // dynamo omits the dtype default (None) from aten::zeros/ones, leaving
        // [size device pin_memory] which matches neither the level-2 fold nor
        // the torch.zeros rewrite; restore the canonical [size dtype device
        // pin_memory] order so the constant folds to an Attribute
        if (!has_input_name(inputnames, "dtype"))
            add_const("dtype", Parameter());
        reorder_inputs({"self", "size", "dtype", "layout", "device", "pin_memory"});
    }
    else if (type == "aten::full")
    {
        // same dtype-default omission for aten::full; restore
        // [size fill_value dtype device pin_memory] for the level-2 fold
        if (!has_input_name(inputnames, "dtype"))
            add_const("dtype", Parameter());
        reorder_inputs({"self", "size", "fill_value", "dtype", "layout", "device", "pin_memory"});
    }
    else if (type == "aten::new_full")
    {
        // Tensor.new_full(self, size, fill_value) omits the dtype/layout/
        // device defaults; restore the canonical order for the level-2 fold
        if (!has_input_name(inputnames, "dtype"))
            add_const("dtype", Parameter());
        if (!has_input_name(inputnames, "layout"))
            add_const("layout", Parameter());
        if (!has_input_name(inputnames, "device"))
            add_const("device", Parameter());
        reorder_inputs({"self", "size", "fill_value", "dtype", "layout", "device", "pin_memory"});
    }
    else if (type == "aten::new_zeros" || type == "aten::new_ones" || type == "aten::new_empty")
    {
        // dynamo emits Tensor.new_zeros(self, size, pin_memory); the pnnx
        // pass_level2 pattern expects input size dtype layout device pin_memory.
        // GraphRewriter matches constant inputs POSITIONALLY, so the final
        // order must be exactly: input size dtype layout device pin_memory.
        std::vector<Operand*> old_inputs = op->inputs;
        std::vector<std::string> old_names = op->inputnames;

        // detach op from all old inputs; keep the interesting ones
        for (size_t j = 0; j < old_inputs.size(); j++)
        {
            auto& cons = old_inputs[j]->consumers;
            cons.erase(std::find(cons.begin(), cons.end(), op));
        }

        op->inputs.clear();
        op->inputnames.clear();

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
            op->inputs.push_back(self_op);
            op->inputnames.push_back("input");
            self_op->consumers.push_back(op);
        }
        if (size_op)
        {
            op->inputs.push_back(size_op);
            op->inputnames.push_back("size");
            size_op->consumers.push_back(op);
        }
        // (size is always present for new_*; if it were missing the pattern
        //  simply will not match and the op stays as-is, which is safe)
        if (dtype_op)
        {
            // keep an explicit dtype constant when dynamo emitted one
            // (e.g. x.new_empty(..., dtype=torch.long)); otherwise null means
            // "inherit self's dtype", matching the pattern default
            op->inputs.push_back(dtype_op);
            op->inputnames.push_back("dtype");
            dtype_op->consumers.push_back(op);
        }
        else
        {
            add_const("dtype", Parameter());
        }
        if (layout_op)
        {
            op->inputs.push_back(layout_op);
            op->inputnames.push_back("layout");
            layout_op->consumers.push_back(op);
        }
        else
        {
            add_const("layout", Parameter());
        }
        if (device_op)
        {
            op->inputs.push_back(device_op);
            op->inputnames.push_back("device");
            device_op->consumers.push_back(op);
        }
        else
        {
            add_const("device", Parameter());
        }
        add_const("pin_memory", have_pin); // fresh constant either way
    }
    else if (type == "aten::ones_like" || type == "aten::zeros_like"
             || type == "aten::rand_like" || type == "aten::randn_like"
             || type == "aten::empty_like" || type == "aten::full_like")
    {
        // dynamo emits input [dtype|fill_value] pin_memory; the pnnx pattern
        // expects input dtype layout device requires_grad memory_format
        // (full_like additionally carries fill_value)
        std::vector<Operand*> old_inputs = op->inputs;
        std::vector<std::string> old_names = op->inputnames;
        for (size_t j = 0; j < old_names.size(); j++)
        {
            if (old_names[j] != "self" && old_names[j] != "input" && old_names[j] != "dtype")
            {
                // drop irrelevant inputs like pin_memory and clean up consumer refs
                auto& cons = old_inputs[j]->consumers;
                cons.erase(std::find(cons.begin(), cons.end(), op));
            }
        }

        op->inputs.clear();
        op->inputnames.clear();

        int found_input = -1;
        for (size_t j = 0; j < old_names.size(); j++)
            if (old_names[j] == "self" || old_names[j] == "input")
            {
                found_input = (int)j;
                break;
            }
        if (found_input != -1)
        {
            op->inputs.push_back(old_inputs[found_input]);
            op->inputnames.push_back("input");
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
            op->inputs.push_back(old_inputs[found_dtype]);
            op->inputnames.push_back("dtype");
        }
        else
        {
            add_const("dtype", Parameter());
        }

        // full_like: dynamo passes fill_value as a scalar input; the pnnx
        // pattern wants it as the first input (input fill_value dtype ...)
        if (type == "aten::full_like")
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
                op->inputs.insert(op->inputs.begin() + 1, fill);
                op->inputnames.insert(op->inputnames.begin() + 1, "fill_value");
            }
            else
            {
                // fill_value missing: synthesize a constant 0 and place it at
                // index 1 (new_constant appends at the end)
                add_const("fill_value", 0);
                Operand* fill = op->inputs.back();
                op->inputs.pop_back();
                op->inputs.insert(op->inputs.begin() + 1, fill);
                op->inputnames.pop_back();
                op->inputnames.insert(op->inputnames.begin() + 1, "fill_value");
            }
        }

        add_const("layout", Parameter());
        add_const("device", Parameter());
        add_const("requires_grad", false);
        add_const("memory_format", Parameter());
    }
}

} // namespace pnnx
