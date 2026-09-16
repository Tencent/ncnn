// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_unsqueeze_transpose_squeeze.h"

#include <algorithm>

#include "ir.h"

namespace pnnx {

void fuse_unsqueeze_transpose_squeeze(Graph& g)
{
    for (int i = (int)g.ops.size() - 1; i >= 0; i--)
    {
        Operator* u = g.ops[i];

        // torch.unsqueeze(dim=u)
        if (u->type != "torch.unsqueeze")
            continue;
        if (u->inputs.size() != 1 || u->outputs.size() != 1)
            continue;
        if (u->params.find("dim") == u->params.end())
            continue;

        int unsq_dim = u->params.at("dim").i;

        Operand* ou = u->outputs[0];
        if (ou->consumers.size() != 1)
            continue;

        Operator* t = ou->consumers[0];
        // torch.transpose(dim0, dim1)
        if (t->type != "torch.transpose")
            continue;
        if (t->inputs.size() != 1 || t->outputs.size() != 1)
            continue;
        if (t->params.find("dim0") == t->params.end() || t->params.find("dim1") == t->params.end())
            continue;

        int td0 = t->params.at("dim0").i;
        int td1 = t->params.at("dim1").i;

        Operand* ot = t->outputs[0];
        if (ot->consumers.size() != 1)
            continue;

        Operator* s = ot->consumers[0];
        // torch.squeeze(dim)
        if (s->type != "torch.squeeze")
            continue;
        if (s->inputs.size() != 1 || s->outputs.size() != 1)
            continue;
        if (s->params.find("dim") == s->params.end())
            continue;

        int sq_dim = s->params.at("dim").i;

        // the input must be 4D
        const std::vector<int>& in_shape = u->inputs[0]->shape;
        if (in_shape.size() != 4)
            continue;

        // intermediate ranks: 5D after unsqueeze, 5D through transpose, 4D after squeeze
        const int r5 = 5;

        // normalize dims (in 5D space)
        if (unsq_dim < 0) unsq_dim += r5;
        if (td0 < 0) td0 += r5;
        if (td1 < 0) td1 += r5;
        if (sq_dim < 0) sq_dim += r5;

        if (unsq_dim < 0 || unsq_dim >= r5) continue;
        if (td0 < 0 || td0 >= r5 || td1 < 0 || td1 >= r5) continue;
        if (sq_dim < 0 || sq_dim >= r5) continue;

        // 5D layout: orig5[pos] stores the original 4D dim number (-1 at the unsqueeze insert)
        std::vector<int> orig5(r5);
        for (int p = 0; p < r5; p++)
        {
            if (p == unsq_dim)
                orig5[p] = -1;
            else if (p < unsq_dim)
                orig5[p] = p;
            else
                orig5[p] = p - 1;
        }

        // transpose swap
        std::swap(orig5[td0], orig5[td1]);

        // the squeezed dim must be the inserted -1 (size-1 dim)
        if (orig5[sq_dim] != -1)
            continue;

        // permute over the remaining 4D dims
        std::vector<int> perm;
        for (int p = 0; p < r5; p++)
        {
            if (p == sq_dim)
                continue;
            if (orig5[p] < 0)
            {
                perm.clear();
                break;
            }
            perm.push_back(orig5[p]);
        }

        if (perm.size() != 4)
            continue;

        // check perm is a permutation of 0..3
        std::vector<int> sorted = perm;
        std::sort(sorted.begin(), sorted.end());
        bool ok = true;
        for (int k = 0; k < 4; k++)
        {
            if (sorted[k] != k) ok = false;
        }
        if (!ok)
            continue;

        // replace: unsqueeze -> Tensor.permute, rewire inputs, delete transpose/squeeze
        Operand* final_out = s->outputs[0];

        u->type = "Tensor.permute";
        u->params.clear();
        u->params["dims"] = perm;

        // update consumers of the intermediate operands
        for (Operand* c : t->inputs)
        {
            c->consumers.erase(std::find(c->consumers.begin(), c->consumers.end(), t));
        }
        for (Operand* c : s->inputs)
        {
            c->consumers.erase(std::find(c->consumers.begin(), c->consumers.end(), s));
        }

        // rewire the unsqueeze output operand to final_out
        ou->consumers.clear();
        ou->producer = 0;
        final_out->producer = u;
        u->outputs.clear();
        u->outputs.push_back(final_out);

        // the intermediate operand ot is managed by the graph; do not delete it
        // manually (avoid double free)
        ot->producer = 0;
        ot->consumers.clear();
        // orphaned operand without producer/consumer is skipped when saving

        g.ops.erase(std::find(g.ops.begin(), g.ops.end(), t));
        delete t;
        g.ops.erase(std::find(g.ops.begin(), g.ops.end(), s));
        delete s;
    }
}

} // namespace pnnx
