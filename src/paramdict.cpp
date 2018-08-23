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
#include "platform.h"
#include <sstream>

namespace ncnn {

ParamDict::ParamDict()
{
    clear();
}

int ParamDict::get(int id, int def) const
{
    return params[id].loaded ? params[id].i : def;
}

float ParamDict::get(int id, float def) const
{
    return params[id].loaded ? params[id].f : def;
}

Mat ParamDict::get(int id, const Mat& def) const
{
    return params[id].loaded ? params[id].v : def;
}

void ParamDict::set(int id, int i)
{
    params[id].loaded = 1;
    params[id].i = i;
}

void ParamDict::set(int id, float f)
{
    params[id].loaded = 1;
    params[id].f = f;
}

void ParamDict::set(int id, const Mat& v)
{
    params[id].loaded = 1;
    params[id].v = v;
}

void ParamDict::clear()
{
    for (int i = 0; i < NCNN_MAX_PARAM_COUNT; i++)
    {
        params[i].loaded = 0;
        params[i].v = Mat();
    }
}

static int read_v_str(const std::string &vstr, int &iValue, float &fValue)
{
    bool is_float = false;
    // look ahead for determine isfloat
    for (int j=0; j<vstr.size(); j++)
    {
        if (vstr[j] == '\0')
            break;

        if (vstr[j] == '.' || tolower(vstr[j]) == 'e') {
            is_float = true;
            break;
        }
    }
    int nscan = 0;

    if (is_float)
        nscan = sscanf(vstr.c_str(), "%f", &fValue);
    else
        nscan = sscanf(vstr.c_str(), "%d", &iValue);
    if (nscan != 1)
    {
        fprintf(stderr, "ParamDict parse value fail\n");
        return -1;
    }
    return 0;
}

int ParamDict::load_param(std::stringstream &ss)
{
    clear();

//     0=100 1=1.250000 -23303=5,0.1,0.2,0.4,0.8,1.0

    // parse each key=value pair
    while (true)
    {
        char c;
        int id = 0;
        ss >>id;
        if (!ss.good()) {
            break;
        };
        ss.get(c);
        if (!ss.good() || c != '=') {
            break;
        }
        bool is_array = id <= -23300;
        if (is_array)
        {
            id = -id - 23300;
        }

        if (is_array)
        {
            int len = 0;
            ss >> len;
            if (!ss.good())
            {
                fprintf(stderr, "ParamDict read array length fail\n");
                return -1;
            }

            params[id].v.create(len);

            std::string array_vstr;
            std::getline(ss, array_vstr, ' ');
            array_vstr.push_back(',');
            std::stringstream vss(array_vstr);
            for (int j = 0; j < len; j++)
            {
                std::string vstr;
                std::getline(vss, vstr, ',');
                if (read_v_str(vstr, ((int*)params[id].v.data)[j], ((float*)params[id].v.data)[j]) < 0) {
                    return -1;
                }
            }
        }
        else
        {
            std::string vstr;
            std::getline(ss, vstr, ' ');
            // fprintf(stderr, "vstr:%s", vstr.c_str());
            if (!ss.good())
            {
                fprintf(stderr, "ParamDict read value fail\n");
                return -1;
            }
            if (read_v_str(vstr, params[id].i, params[id].f) < 0) {
                return -1;
            }
        }

        params[id].loaded = 1;
    }

    return 0;
}

#if NCNN_STDIO

int ParamDict::load_param_bin(FILE* fp)
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
    fread(&id, sizeof(int), 1, fp);

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
            fread(&len, sizeof(int), 1, fp);

            params[id].v.create(len);

            float* ptr = params[id].v;
            fread(ptr, sizeof(float), len, fp);
        }
        else
        {
            fread(&params[id].f, sizeof(float), 1, fp);
        }

        params[id].loaded = 1;

        fread(&id, sizeof(int), 1, fp);
    }

    return 0;
}
#endif // NCNN_STDIO

int ParamDict::load_param(const unsigned char*& mem)
{
    clear();

    int id = *(int*)(mem);
    mem += 4;

    while (id != -233)
    {
        bool is_array = id <= -23300;
        if (is_array)
        {
            id = -id - 23300;
        }

        if (is_array)
        {
            int len = *(int*)(mem);
            mem += 4;

            params[id].v.create(len);

            memcpy(params[id].v.data, mem, len * 4);
            mem += len * 4;
        }
        else
        {
            params[id].f = *(float*)(mem);
            mem += 4;
        }

        params[id].loaded = 1;

        id = *(int*)(mem);
        mem += 4;
    }

    return 0;
}

} // namespace ncnn
