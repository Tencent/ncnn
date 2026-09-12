// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_half_to_float.h"

#include <string.h>

namespace pnnx {

namespace ncnn {

void convert_half_to_float(Graph& graph)
{
    for (Operator* op : graph.ops)
    {
        while (1)
        {
            bool matched = false;

            for (auto x : op->attrs)
            {
                const Attribute& attr = x.second;
                if (attr.type != 3)
                    continue;

                // fp16 -> fp32
                const int ec = attr.elemcount();
                if (ec <= 0)
                    continue; // nothing to convert; skip (keep matched false)

                matched = true;

                Attribute attr_new;
                attr_new.type = 1;
                attr_new.shape = attr.shape;
                attr_new.data.resize((size_t)ec * 4);

                auto p = attr.get_float32_data();
                memcpy((void*)attr_new.data.data(), (const void*)p.data(), std::min(attr_new.data.size(), p.size() * sizeof(float)));

                op->attrs[x.first] = attr_new;

                break;
            }

            if (!matched)
                break;
        }
    }
}

} // namespace ncnn

} // namespace pnnx
