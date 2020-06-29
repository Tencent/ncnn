// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "paramdict.h"

#include "datareader.h"
#include "platform.h"

#include <ctype.h>

#if NCNN_STDIO || NCNN_STRING
#include <stdio.h>
#endif

namespace ncnn {

ParamDict::ParamDict()
{
    clear();
}

// TODO strict type check
int ParamDict::get(int id, int def) const
{
    return params[id].type ? params[id].i : def;
}

float ParamDict::get(int id, float def) const
{
    return params[id].type ? params[id].f : def;
}

Mat ParamDict::get(int id, const Mat& def) const
{
    return params[id].type ? params[id].v : def;
}

void ParamDict::set(int id, int i)
{
    params[id].type = 2;
    params[id].i = i;
}

void ParamDict::set(int id, float f)
{
    params[id].type = 3;
    params[id].f = f;
}

void ParamDict::set(int id, const Mat& v)
{
    params[id].type = 4;
    params[id].v = v;
}

void ParamDict::clear()
{
    for (int i = 0; i < NCNN_MAX_PARAM_COUNT; i++)
    {
        params[i].type = 0;
        params[i].v = Mat();
    }
}

#if NCNN_STRING
static bool vstr_is_float(const char vstr[16])
{
    // look ahead for determine isfloat
    for (int j = 0; j < 16; j++)
    {
        if (vstr[j] == '\0')
            break;

        if (vstr[j] == '.' || tolower(vstr[j]) == 'e')
            return true;
    }

    return false;
}

static float vstr_to_float(const char vstr[16])
{
    double v = 0.0;

    const char* p = vstr;

    // sign
    bool sign = *p != '-';
    if (*p == '+' || *p == '-')
    {
        p++;
    }

    // digits before decimal point or exponent
    unsigned int v1 = 0;
    while (isdigit(*p))
    {
        v1 = v1 * 10 + (*p - '0');
        p++;
    }

    v = (double)v1;

    // digits after decimal point
    if (*p == '.')
    {
        p++;

        unsigned int pow10 = 1;
        unsigned int v2 = 0;

        while (isdigit(*p))
        {
            v2 = v2 * 10 + (*p - '0');
            pow10 *= 10;
            p++;
        }

        v += v2 / (double)pow10;
    }

    // exponent
    if (*p == 'e' || *p == 'E')
    {
        p++;

        // sign of exponent
        bool fact = *p != '-';
        if (*p == '+' || *p == '-')
        {
            p++;
        }

        // digits of exponent
        unsigned int expon = 0;
        while (isdigit(*p))
        {
            expon = expon * 10 + (*p - '0');
            p++;
        }

        double scale = 1.0;
        while (expon >= 8)
        {
            scale *= 1e8;
            expon -= 8;
        }
        while (expon > 0)
        {
            scale *= 10.0;
            expon -= 1;
        }

        v = fact ? v * scale : v / scale;
    }

    //     fprintf(stderr, "v = %f\n", v);
    return sign ? (float)v : (float)-v;
}

int ParamDict::load_param(const DataReader& dr)
{
    clear();

    //     0=100 1=1.250000 -23303=5,0.1,0.2,0.4,0.8,1.0

    // parse each key=value pair
    int id = 0;
    while (dr.scan("%d=", &id) == 1)
    {
        bool is_array = id <= -23300;
        if (is_array)
        {
            id = -id - 23300;
        }

        if (is_array)
        {
            int len = 0;
            int nscan = dr.scan("%d", &len);
            if (nscan != 1)
            {
                NCNN_LOGE("ParamDict read array length failed");
                return -1;
            }

            params[id].v.create(len);

            for (int j = 0; j < len; j++)
            {
                char vstr[16];
                nscan = dr.scan(",%15[^,\n ]", vstr);
                if (nscan != 1)
                {
                    NCNN_LOGE("ParamDict read array element failed");
                    return -1;
                }

                bool is_float = vstr_is_float(vstr);

                if (is_float)
                {
                    float* ptr = params[id].v;
                    ptr[j] = vstr_to_float(vstr);
                }
                else
                {
                    int* ptr = params[id].v;
                    nscan = sscanf(vstr, "%d", &ptr[j]);
                    if (nscan != 1)
                    {
                        NCNN_LOGE("ParamDict parse array element failed");
                        return -1;
                    }
                }

                params[id].type = is_float ? 6 : 5;
            }
        }
        else
        {
            char vstr[16];
            int nscan = dr.scan("%15s", vstr);
            if (nscan != 1)
            {
                NCNN_LOGE("ParamDict read value failed");
                return -1;
            }

            bool is_float = vstr_is_float(vstr);

            if (is_float)
            {
                params[id].f = vstr_to_float(vstr);
            }
            else
            {
                nscan = sscanf(vstr, "%d", &params[id].i);
                if (nscan != 1)
                {
                    NCNN_LOGE("ParamDict parse value failed");
                    return -1;
                }
            }

            params[id].type = is_float ? 3 : 2;
        }
    }

    return 0;
}
#endif // NCNN_STRING

int ParamDict::load_param_bin(const DataReader& dr)
{
    clear();

    //     binary 0
    //     binary 100
    //     binary 1
    //     binary 1.250000
    //     binary 3 | array_bit
    //     binary 5
    //     binary 0.1
    //     binary 0.2
    //     binary 0.4
    //     binary 0.8
    //     binary 1.0
    //     binary -233(EOP)

    int id = 0;
    size_t nread;
    nread = dr.read(&id, sizeof(int));
    if (nread != sizeof(int))
    {
        NCNN_LOGE("ParamDict read id failed %zd", nread);
        return -1;
    }

    while (id != -233)
    {
        bool is_array = id <= -23300;
        if (is_array)
        {
            id = -id - 23300;
        }

        if (is_array)
        {
            int len = 0;
            nread = dr.read(&len, sizeof(int));
            if (nread != sizeof(int))
            {
                NCNN_LOGE("ParamDict read array length failed %zd", nread);
                return -1;
            }

            params[id].v.create(len);

            float* ptr = params[id].v;
            nread = dr.read(ptr, sizeof(float) * len);
            if (nread != sizeof(float) * len)
            {
                NCNN_LOGE("ParamDict read array element failed %zd", nread);
                return -1;
            }

            params[id].type = 4;
        }
        else
        {
            nread = dr.read(&params[id].f, sizeof(float));
            if (nread != sizeof(float))
            {
                NCNN_LOGE("ParamDict read value failed %zd", nread);
                return -1;
            }

            params[id].type = 1;
        }

        nread = dr.read(&id, sizeof(int));
        if (nread != sizeof(int))
        {
            NCNN_LOGE("ParamDict read EOP failed %zd", nread);
            return -1;
        }
    }

    return 0;
}

} // namespace ncnn
