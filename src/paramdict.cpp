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
#include <stdarg.h>
#include <stdio.h>
#include "paramdict.h"
#include "platform.h"

namespace ncnn {

ParamDict::ParamDict()
{
    use_winograd_convolution = 1;
    use_sgemm_convolution = 1;
    use_int8_inference = 1;

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

#if NCNN_STDIO
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

int ParamDict::load_param(FILE* fp)
{
    clear();

//     0=100 1=1.250000 -23303=5,0.1,0.2,0.4,0.8,1.0

    // parse each key=value pair
    int id = 0;
    while (fscanf(fp, "%d=", &id) == 1)
    {
        bool is_array = id <= -23300;
        if (is_array)
        {
            id = -id - 23300;
        }

        if (is_array)
        {
            int len = 0;
            int nscan = fscanf(fp, "%d", &len);
            if (nscan != 1)
            {
                fprintf(stderr, "ParamDict read array length fail\n");
                return -1;
            }

            params[id].v.create(len);

            for (int j = 0; j < len; j++)
            {
                char vstr[16];
                nscan = fscanf(fp, ",%15[^,\n ]", vstr);
                if (nscan != 1)
                {
                    fprintf(stderr, "ParamDict read array element fail\n");
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
                    fprintf(stderr, "ParamDict parse array element fail\n");
                    return -1;
                }
            }
        }
        else
        {
            char vstr[16];
            int nscan = fscanf(fp, "%15s", vstr);
            if (nscan != 1)
            {
                fprintf(stderr, "ParamDict read value fail\n");
                return -1;
            }

            bool is_float = vstr_is_float(vstr);

            if (is_float)
                nscan = sscanf(vstr, "%f", &params[id].f);
            else
                nscan = sscanf(vstr, "%d", &params[id].i);
            if (nscan != 1)
            {
                fprintf(stderr, "ParamDict parse value fail\n");
                return -1;
            }
        }

        params[id].loaded = 1;
    }

    return 0;
}

#if _MSC_VER
static inline int mem_sscanf_with_n(int* _internal_nconsumed_ptr, const char*& ptr, const char* format, ...)
{
    va_list args;
    va_start(args, format);

    int _n = vsscanf(ptr, format, args);

    va_end(args);

    ptr += *_internal_nconsumed_ptr;

    return *_internal_nconsumed_ptr > 0 ? _n : 0;
}
#define mem_sscanf(ptr, format, ...)  mem_sscanf_with_n(&_internal_nconsumed, ptr, format "%n", __VA_ARGS__, &_internal_nconsumed)
#else
// return value from macro requires gcc extension https://gcc.gnu.org/onlinedocs/gcc/Statement-Exprs.html
#define mem_sscanf(ptr, format, ...)  ({int _b=0; int _n = sscanf(ptr, format "%n", __VA_ARGS__, &_b); ptr+=_b;_b>0?_n:0;})
#endif // _MSC_VER

int ParamDict::load_param_mem(const char*& mem)
{
#if _MSC_VER
    int _internal_nconsumed;
#endif

    clear();

//     0=100 1=1.250000 -23303=5,0.1,0.2,0.4,0.8,1.0

    // parse each key=value pair
    int id = 0;
    while (mem_sscanf(mem, "%d=", &id) == 1)
    {
        bool is_array = id <= -23300;
        if (is_array)
        {
            id = -id - 23300;
        }

        if (is_array)
        {
            int len = 0;
            int nscan = mem_sscanf(mem, "%d", &len);
            if (nscan != 1)
            {
                fprintf(stderr, "ParamDict read array length fail\n");
                return -1;
            }

            params[id].v.create(len);

            for (int j = 0; j < len; j++)
            {
                char vstr[16];
                nscan = mem_sscanf(mem, ",%15[^,\n ]", vstr);
                if (nscan != 1)
                {
                    fprintf(stderr, "ParamDict read array element fail\n");
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
                    fprintf(stderr, "ParamDict parse array element fail\n");
                    return -1;
                }
            }
        }
        else
        {
            char vstr[16];
            int nscan = mem_sscanf(mem, "%15s", vstr);
            if (nscan != 1)
            {
                fprintf(stderr, "ParamDict read value fail\n");
                return -1;
            }

            bool is_float = vstr_is_float(vstr);

            if (is_float)
                nscan = sscanf(vstr, "%f", &params[id].f);
            else
                nscan = sscanf(vstr, "%d", &params[id].i);
            if (nscan != 1)
            {
                fprintf(stderr, "ParamDict parse value fail\n");
                return -1;
            }
        }

        params[id].loaded = 1;
    }
    return 0;
}
#endif // NCNN_STRING

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
