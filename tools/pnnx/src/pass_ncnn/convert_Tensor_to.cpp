// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_Tensor_to.h"

#include <algorithm>

namespace pnnx {

namespace ncnn {

static bool is_shape_only_operator(const std::string& t)
{
    return t == "aten::unsqueeze" || t == "torch.unsqueeze" || t == "aten::expand" || t == "Tensor.expand" || t == "aten::reshape" || t == "Tensor.reshape" || t == "aten::view" || t == "Tensor.view" || t == "aten::permute" || t == "Tensor.permute" || t == "aten::transpose" || t == "Tensor.transpose" || t == "aten::contiguous" || t == "Tensor.contiguous" || t == "aten::squeeze" || t == "Tensor.squeeze" || t == "aten::flatten" || t == "Tensor.flatten";
}

void convert_Tensor_to(Graph& graph)
{
    for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
    {
        Operator* op = graph.ops[i];

        if (op->type != "Tensor.to")
            continue;
        if (op->inputs.size() != 1 || op->outputs.size() != 1)
            continue;
        if (!op->has_param("dtype"))
            continue;

        // only handle to-f32
        const Parameter& dt = op->params.at("dtype");
        if (!(dt.type == 4 && dt.s == "torch.float") && !(dt.type == 2 && dt.i == 1))
            continue;

        Operand* in = op->inputs[0];
        Operand* out = op->outputs[0];
        Operand* tail = in; // chain tail (keeps the to input shape)

        // walk back along the shape-only chain to find an integer Attribute
        std::vector<Operator*> chain;
        Operator* cur = in->producer;
        while (cur && is_shape_only_operator(cur->type) && cur->inputs.size() == 1)
        {
            chain.push_back(cur);
            in = cur->inputs[0];
            cur = in->producer;
        }

        // the chain origin must be an integer pnnx.Attribute
        Operator* attr_op = in->producer;
        if (!attr_op || attr_op->type != "pnnx.Attribute")
            continue;
        if (attr_op->attrs.find("data") == attr_op->attrs.end())
            continue;
        Attribute& a = attr_op->attrs.at("data");
        if (a.type != 5 && a.type != 4 && a.type != 6 && a.type != 7 && a.type != 8)
            continue; // only i64/i32/i16/i8/u8

        // convert integer Attribute data to f32 (small integers like position ids are exact in f32)
        int count = (int)a.data.size() / (int)a.elemsize();
        if (count <= 0)
            continue;
        std::vector<char> fdata(count * 4);
        float* fp = (float*)fdata.data();
        if (a.type == 5) // i64
        {
            const int64_t* p = (const int64_t*)a.data.data();
            for (int j = 0; j < count; j++)
                fp[j] = (float)p[j];
        }
        else if (a.type == 4) // i32
        {
            const int* p = (const int*)a.data.data();
            for (int j = 0; j < count; j++)
                fp[j] = (float)p[j];
        }
        else if (a.type == 6) // i16
        {
            const short* p = (const short*)a.data.data();
            for (int j = 0; j < count; j++)
                fp[j] = (float)p[j];
        }
        else if (a.type == 7) // i8
        {
            const signed char* p = (const signed char*)a.data.data();
            for (int j = 0; j < count; j++)
                fp[j] = (float)p[j];
        }
        else if (a.type == 8) // u8
        {
            const unsigned char* p = (const unsigned char*)a.data.data();
            for (int j = 0; j < count; j++)
                fp[j] = (float)p[j];
        }
        else
        {
            continue;
        }

        // if any shape-only chain node's output is consumed beyond the next
        // chain node / the to op, relabeling it f32 (or the shared attribute)
        // would corrupt those other consumers; refuse to rewrite in that case.
        // this must be checked regardless of how many consumers the attribute
        // itself has: an intermediate such as unsqueeze may branch to another
        // consumer (e.g. embedding indices) even when the attribute feeds a
        // single chain.
        bool chain_shared = false;
        for (Operator* c : chain)
        {
            for (Operand* o : c->outputs)
            {
                if (o->consumers.size() > 1)
                    chain_shared = true;
            }
        }
        if (chain_shared)
            continue;

        // if the integer attribute is shared with other consumers, clone a
        // private f32 copy for this to-f32 chain instead of mutating the shared
        // attribute in place (other consumers would see the changed dtype/bytes)
        if (in->consumers.size() > 1)
        {
            // the clone must be wired at the origin-facing node of the chain:
            // the one that directly consumes the attribute output. when the
            // chain is non-empty that is chain.back() (the walk-back ends with
            // in == chain.back()->inputs[0]); chain.front() is the to-facing
            // end and does not consume the attribute.
            Operator* origin_consumer = chain.empty() ? op : chain.back();

            Operator* clone_op = graph.new_operator_before("pnnx.Attribute", attr_op->name + "_to_f32", origin_consumer);
            Operand* clone_out = graph.new_operand(clone_op->name + "_out");
            clone_op->outputs.push_back(clone_out);
            clone_out->producer = clone_op;
            clone_out->type = 1;
            clone_out->shape = a.shape;

            Attribute fattr = a;
            fattr.type = 1;
            fattr.data = fdata;
            clone_op->attrs["data"] = fattr;

            for (size_t j = 0; j < in->consumers.size(); j++)
            {
                if (in->consumers[j] == origin_consumer)
                {
                    in->consumers.erase(in->consumers.begin() + j);
                    break;
                }
            }
            for (size_t j = 0; j < origin_consumer->inputs.size(); j++)
            {
                if (origin_consumer->inputs[j] == in)
                    origin_consumer->inputs[j] = clone_out;
            }
            clone_out->consumers.push_back(origin_consumer);

            in = clone_out;
            // empty chain: tail was the original (still-int) attribute output, so
            // the to-f32 consumers must instead consume the f32 clone
            if (chain.empty())
                tail = clone_out;
        }
        else
        {
            a.type = 1; // f32
            a.data = fdata;
        }

        // mark all outputs on the chain as f32
        for (Operator* c : chain)
        {
            for (Operand* o : c->outputs)
                o->type = 1;
        }
        tail->type = 1;
        in->type = 1;

        // reconnect: consumers of the to output now use tail (chain tail, now f32, keeps to input shape); remove to
        for (Operator* c : out->consumers)
        {
            for (size_t j = 0; j < c->inputs.size(); j++)
            {
                if (c->inputs[j] == out)
                    c->inputs[j] = tail;
            }
            if (std::find(tail->consumers.begin(), tail->consumers.end(), c) == tail->consumers.end())
                tail->consumers.push_back(c);
        }
        out->producer = 0;
        out->consumers.clear();

        graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), out));
        delete out;

        // remove op from the consumers of its inputs to avoid dangling pointers
        for (Operand* in_op : op->inputs)
        {
            auto it = std::find(in_op->consumers.begin(), in_op->consumers.end(), op);
            if (it != in_op->consumers.end())
                in_op->consumers.erase(it);
        }
        op->inputs.clear();
        op->outputs.clear();
        graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op));
        delete op;
    }
}

} // namespace ncnn

} // namespace pnnx
