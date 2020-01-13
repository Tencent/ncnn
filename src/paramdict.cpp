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

#include <ctype.h>
#include "paramdict.h"
#include "datareader.h"
#include "platform.h"

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
    for (int j=0; j<16; j++)
    {
        if (vstr[j] == '\0')
            break;

        if (vstr[j] == '.' || tolower(vstr[j]) == 'e')
            return true;
    }

    return false;
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
                fprintf(stderr, "ParamDict read array length failed\n");
                return -1;
            }

            params[id].v.create(len);

            for (int j = 0; j < len; j++)
            {
                char vstr[16];
                nscan = dr.scan(",%15[^,\n ]", vstr);
                if (nscan != 1)
                {
                    fprintf(stderr, "ParamDict read array element failed\n");
                    return -1;
                }

                bool is_float = vstr_is_float(vstr);

                if (is_float)
                {
                    float* ptr = params[id].v;
                    nscan = sscanf(vstr, "%f", &ptr[j]);
                }
                else
                {
                    int* ptr = params[id].v;
                    nscan = sscanf(vstr, "%d", &ptr[j]);
                }
                if (nscan != 1)
                {
                    fprintf(stderr, "ParamDict parse array element failed\n");
                    return -1;
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
                fprintf(stderr, "ParamDict read value failed\n");
                return -1;
            }

            bool is_float = vstr_is_float(vstr);

            if (is_float)
                nscan = sscanf(vstr, "%f", &params[id].f);
            else
                nscan = sscanf(vstr, "%d", &params[id].i);
            if (nscan != 1)
            {
                fprintf(stderr, "ParamDict parse value failed\n");
                return -1;
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
        fprintf(stderr, "ParamDict read id failed %zd\n", nread);
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
                fprintf(stderr, "ParamDict read array length failed %zd\n", nread);
                return -1;
            }

            params[id].v.create(len);

            float* ptr = params[id].v;
            nread = dr.read(ptr, sizeof(float) * len);
            if (nread != sizeof(float) * len)
            {
                fprintf(stderr, "ParamDict read array element failed %zd\n", nread);
                return -1;
            }

            params[id].type = 4;
        }
        else
        {
            nread = dr.read(&params[id].f, sizeof(float));
            if (nread != sizeof(float))
            {
                fprintf(stderr, "ParamDict read value failed %zd\n", nread);
                return -1;
            }

            params[id].type = 1;
        }

        nread = dr.read(&id, sizeof(int));
        if (nread != sizeof(int))
        {
            fprintf(stderr, "ParamDict read EOP failed %zd\n", nread);
            return -1;
        }
    }

    return 0;
}

} // namespace ncnn
