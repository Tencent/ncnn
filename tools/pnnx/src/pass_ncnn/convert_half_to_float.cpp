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

                matched = true;

                // fp16 -> fp32
                Attribute attr_new;
                attr_new.type = 1;
                attr_new.shape = attr.shape;
                attr_new.data.resize(attr.elemcount() * 4);

                auto p = attr.get_float32_data();
                memcpy((void*)attr_new.data.data(), (const void*)p.data(), attr_new.data.size());

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
