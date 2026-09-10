// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_to_float.h"

#include <stdint.h>
#include <string.h>

namespace pnnx {

namespace ncnn {

void convert_to_float(Graph& graph)
{
    for (Operator* op : graph.ops)
    {
        while (1)
        {
            bool matched = false;

            for (auto x : op->attrs)
            {
                const Attribute& attr = x.second;
                if (attr.type != 2 && attr.type != 3 && attr.type != 13)
                    continue;

                matched = true;

                // fp16/fp64/bf16 -> fp32
                Attribute attr_new;
                attr_new.type = 1;
                attr_new.shape = attr.shape;
                attr_new.data.resize((size_t)attr.elemcount() * 4);

                if (attr.type == 13)
                {
                    for (size_t i = 0; i < attr_new.data.size() / 4; i++)
                    {
                        uint16_t value;
                        memcpy(&value, attr.data.data() + i * 2, 2);
                        const uint32_t bits = (uint32_t)value << 16;
                        memcpy(attr_new.data.data() + i * 4, &bits, 4);
                    }
                }
                else
                {
                    auto p = attr.get_float32_data();
                    memcpy((void*)attr_new.data.data(), (const void*)p.data(), attr_new.data.size());
                }

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
