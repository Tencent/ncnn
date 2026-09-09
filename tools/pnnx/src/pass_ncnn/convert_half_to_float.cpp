// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_half_to_float.h"

#include <string.h>

namespace pnnx {

namespace ncnn {

static float bfloat16_to_float32(unsigned short value)
{
    unsigned int bits = (unsigned int)value << 16;
    float value_fp32;
    memcpy(&value_fp32, &bits, sizeof(value_fp32));
    return value_fp32;
}

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
                if (attr.type != 3 && attr.type != 13)
                    continue;

                matched = true;

                Attribute attr_new;
                attr_new.type = 1;
                attr_new.shape = attr.shape;
                const size_t elemcount = attr.elemcount();
                attr_new.data.resize(elemcount * 4);

                if (attr.type == 3)
                {
                    auto p = attr.get_float32_data();
                    memcpy((void*)attr_new.data.data(), (const void*)p.data(), attr_new.data.size());
                }
                else
                {
                    const unsigned short* p = (const unsigned short*)attr.data.data();
                    float* p_new = (float*)attr_new.data.data();
                    for (size_t i = 0; i < elemcount; i++)
                    {
                        p_new[i] = bfloat16_to_float32(p[i]);
                    }
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
