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
    params[id].type = PARAM_TYPE_INT;
    params[id].i = i;
}

void ParamDict::set(int id, float f)
{
    params[id].type = PARAM_TYPE_FLOAT;
    params[id].f = f;
}

void ParamDict::set(int id, const Mat& v)
{
    params[id].type = PARAM_TYPE_FLOAT_MAT;     // TODO: Set it float for the time being
    params[id].v = v;
}

void ParamDict::clear()
{
    for (int i = 0; i < NCNN_MAX_PARAM_COUNT; i++)
    {
        params[i].type = PARAM_TYPE_NULL;
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
                    params[id].type = PARAM_TYPE_FLOAT_MAT;
                }
                else
                {
                    int* ptr = params[id].v;
                    nscan = sscanf(vstr, "%d", &ptr[j]);
                    params[id].type = PARAM_TYPE_INT_MAT;
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

            if (is_float) {
                nscan = sscanf(vstr, "%f", &params[id].f);
                params[id].type = PARAM_TYPE_FLOAT;
            }
            else
            {
                nscan = sscanf(vstr, "%d", &params[id].i);
                params[id].type = PARAM_TYPE_INT;
            }
            if (nscan != 1)
            {
                fprintf(stderr, "ParamDict parse value fail\n");
                return -1;
            }
        }

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
            params[id].type = PARAM_TYPE_FLOAT_MAT;
        }
        else
        {
            fread(&params[id].f, sizeof(float), 1, fp);
            params[id].type = PARAM_TYPE_FLOAT;
        }


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

            params[id].type = PARAM_TYPE_FLOAT_MAT;     // TODO: Set it float for the time being
        }
        else
        {
            params[id].f = *(float*)(mem);
            mem += 4;

            params[id].type = PARAM_TYPE_FLOAT;
        }


        id = *(int*)(mem);
        mem += 4;
    }

    return 0;
}

#if NCNN_SAVER
int ParamDictSaver::save_param(FILE* pp)
{
    for (int id = 0; id < NCNN_MAX_PARAM_COUNT; id++) {
        if (params[id].type == PARAM_TYPE_FLOAT) {
            fprintf(pp, " %d=%f", id, params[id].f);
        } else if (params[id].type == PARAM_TYPE_INT) {
            fprintf(pp, " %d=%d", id, params[id].i);
        } else if (params[id].type == PARAM_TYPE_FLOAT_MAT) {
            int saved_id = -id - 23300;
            size_t len = params[id].v.total();
            fprintf(pp, " %d=%lu", saved_id, len);
            for (size_t i = 0; i < len; i++) {
                fprintf(pp, ",%f", *(static_cast<float *>(params[id].v) + i));
            }
        } else if (params[id].type == PARAM_TYPE_INT_MAT) {
            int saved_id = -id - 23300;
            size_t len = params[id].v.total();
            fprintf(pp, " %d=%lu", saved_id, len);
            for (size_t i = 0; i < len; i++) {
                fprintf(pp, ",%d", *(static_cast<int *>(params[id].v) + i));
            }
        }
    }

    return 0;
}
#endif

} // namespace ncnn
