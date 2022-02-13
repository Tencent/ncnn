// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "convert_half_to_float.h"

namespace pnnx {

namespace ncnn {

static float float16_to_float32(unsigned short value)
{
    // 1 : 5 : 10
    unsigned short sign = (value & 0x8000) >> 15;
    unsigned short exponent = (value & 0x7c00) >> 10;
    unsigned short significand = value & 0x03FF;

    //     NCNN_LOGE("%d %d %d", sign, exponent, significand);

    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;
    if (exponent == 0)
    {
        if (significand == 0)
        {
            // zero
            tmp.u = (sign << 31);
        }
        else
        {
            // denormal
            exponent = 0;
            // find non-zero bit
            while ((significand & 0x200) == 0)
            {
                significand <<= 1;
                exponent++;
            }
            significand <<= 1;
            significand &= 0x3FF;
            tmp.u = (sign << 31) | ((-exponent + (-15 + 127)) << 23) | (significand << 13);
        }
    }
    else if (exponent == 0x1F)
    {
        // infinity or NaN
        tmp.u = (sign << 31) | (0xFF << 23) | (significand << 13);
    }
    else
    {
        // normalized
        tmp.u = (sign << 31) | ((exponent + (-15 + 127)) << 23) | (significand << 13);
    }

    return tmp.f;
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
                if (attr.type != 3)
                    continue;

                matched = true;

                // fp16 -> fp32
                Attribute attr_new;
                attr_new.type = 1;
                attr_new.shape = attr.shape;
                attr_new.data.resize(attr.data.size() * 2);

                const unsigned short* p = (const unsigned short*)attr.data.data();
                float* outp = (float*)attr_new.data.data();
                int len = attr_new.data.size() / 4;
                for (int i = 0; i < len; i++)
                {
                    outp[i] = float16_to_float32(p[i]);
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
