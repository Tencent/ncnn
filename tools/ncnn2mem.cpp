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

#include "layer.h"

#include <cstddef>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

static std::vector<std::string> layer_names;
static std::vector<std::string> blob_names;

static int find_blob_index_by_name(const char* name)
{
    for (std::size_t i = 0; i < blob_names.size(); i++)
    {
        if (blob_names[i] == name)
        {
            return static_cast<int>(i);
        }
    }

    fprintf(stderr, "find_blob_index_by_name %s failed\n", name);
    return -1;
}

static void sanitize_name(char* name)
{
    for (std::size_t i = 0; i < strlen(name); i++)
    {
        if (!isalnum(name[i]))
        {
            name[i] = '_';
        }
    }
}

static std::string path_to_varname(const char* path)
{
    const char* lastslash = strrchr(path, '/');
    const char* name = lastslash == NULL ? path : lastslash + 1;

    std::string varname = name;
    sanitize_name((char*)varname.c_str());

    return varname;
}

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

static int dump_param(const char* parampath, const char* parambinpath, const char* idcpppath)
{
    FILE* fp = fopen(parampath, "rb");

    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", parampath);
        return -1;
    }

    FILE* mp = fopen(parambinpath, "wb");
    FILE* ip = fopen(idcpppath, "wb");

    std::string param_var = path_to_varname(parampath);

    std::string include_guard_var = path_to_varname(idcpppath);

    fprintf(ip, "#ifndef NCNN_INCLUDE_GUARD_%s\n", include_guard_var.c_str());
    fprintf(ip, "#define NCNN_INCLUDE_GUARD_%s\n", include_guard_var.c_str());
    fprintf(ip, "namespace %s_id {\n", param_var.c_str());

    int nscan = 0;
    int magic = 0;
    nscan = fscanf(fp, "%d", &magic);
    if (nscan != 1)
    {
        fprintf(stderr, "read magic failed %d\n", nscan);
        return -1;
    }
    fwrite(&magic, sizeof(int), 1, mp);

    int layer_count = 0;
    int blob_count = 0;
    nscan = fscanf(fp, "%d %d", &layer_count, &blob_count);
    if (nscan != 2)
    {
        fprintf(stderr, "read layer_count and blob_count failed %d\n", nscan);
        return -1;
    }
    fwrite(&layer_count, sizeof(int), 1, mp);
    fwrite(&blob_count, sizeof(int), 1, mp);

    layer_names.resize(layer_count);
    blob_names.resize(blob_count);

    int blob_index = 0;
    for (int i = 0; i < layer_count; i++)
    {
        char layer_type[33];
        char layer_name[257];
        int bottom_count = 0;
        int top_count = 0;
        nscan = fscanf(fp, "%32s %256s %d %d", layer_type, layer_name, &bottom_count, &top_count);
        if (nscan != 4)
        {
            fprintf(stderr, "read layer params failed %d\n", nscan);
            return -1;
        }

        sanitize_name(layer_name);

        int typeindex = ncnn::layer_to_index(layer_type);
        fwrite(&typeindex, sizeof(int), 1, mp);

        fwrite(&bottom_count, sizeof(int), 1, mp);
        fwrite(&top_count, sizeof(int), 1, mp);

        fprintf(ip, "const int LAYER_%s = %d;\n", layer_name, i);

        //         layer->bottoms.resize(bottom_count);
        for (int j = 0; j < bottom_count; j++)
        {
            char bottom_name[257];
            nscan = fscanf(fp, "%256s", bottom_name);
            if (nscan != 1)
            {
                fprintf(stderr, "read bottom_name failed %d\n", nscan);
                return -1;
            }

            sanitize_name(bottom_name);

            int bottom_blob_index = find_blob_index_by_name(bottom_name);

            fwrite(&bottom_blob_index, sizeof(int), 1, mp);
        }

        //         layer->tops.resize(top_count);
        for (int j = 0; j < top_count; j++)
        {
            char blob_name[257];
            nscan = fscanf(fp, "%256s", blob_name);
            if (nscan != 1)
            {
                fprintf(stderr, "read blob_name failed %d\n", nscan);
                return -1;
            }

            sanitize_name(blob_name);

            blob_names[blob_index] = std::string(blob_name);

            fprintf(ip, "const int BLOB_%s = %d;\n", blob_name, blob_index);

            fwrite(&blob_index, sizeof(int), 1, mp);

            blob_index++;
        }

        // dump layer specific params
        // parse each key=value pair
        int id = 0;
        while (fscanf(fp, "%d=", &id) == 1)
        {
            fwrite(&id, sizeof(int), 1, mp);

            bool is_array = id <= -23300;

            if (is_array)
            {
                int len = 0;
                nscan = fscanf(fp, "%d", &len);
                if (nscan != 1)
                {
                    fprintf(stderr, "read array length failed %d\n", nscan);
                    return -1;
                }
                fwrite(&len, sizeof(int), 1, mp);

                for (int j = 0; j < len; j++)
                {
                    char vstr[16];
                    nscan = fscanf(fp, ",%15[^,\n ]", vstr);
                    if (nscan != 1)
                    {
                        fprintf(stderr, "read array element failed %d\n", nscan);
                        return -1;
                    }

                    bool is_float = vstr_is_float(vstr);

                    if (is_float)
                    {
                        float vf;
                        sscanf(vstr, "%f", &vf);
                        fwrite(&vf, sizeof(float), 1, mp);
                    }
                    else
                    {
                        int v;
                        sscanf(vstr, "%d", &v);
                        fwrite(&v, sizeof(int), 1, mp);
                    }
                }
            }
            else
            {
                char vstr[16];
                nscan = fscanf(fp, "%15s", vstr);
                if (nscan != 1)
                {
                    fprintf(stderr, "read value failed %d\n", nscan);
                    return -1;
                }

                bool is_float = vstr_is_float(vstr);

                if (is_float)
                {
                    float vf;
                    sscanf(vstr, "%f", &vf);
                    fwrite(&vf, sizeof(float), 1, mp);
                }
                else
                {
                    int v;
                    sscanf(vstr, "%d", &v);
                    fwrite(&v, sizeof(int), 1, mp);
                }
            }
        }

        int EOP = -233;
        fwrite(&EOP, sizeof(int), 1, mp);

        layer_names[i] = std::string(layer_name);
    }

    fprintf(ip, "} // namespace %s_id\n", param_var.c_str());
    fprintf(ip, "#endif // NCNN_INCLUDE_GUARD_%s\n", include_guard_var.c_str());

    fclose(fp);

    fclose(mp);
    fclose(ip);

    return 0;
}

static int write_memcpp(const char* parambinpath, const char* modelpath, const char* memcpppath)
{
    FILE* cppfp = fopen(memcpppath, "wb");

    // dump param
    std::string param_var = path_to_varname(parambinpath);

    std::string include_guard_var = path_to_varname(memcpppath);

    FILE* mp = fopen(parambinpath, "rb");

    if (!mp)
    {
        fprintf(stderr, "fopen %s failed\n", parambinpath);
        return -1;
    }

    fprintf(cppfp, "#ifndef NCNN_INCLUDE_GUARD_%s\n", include_guard_var.c_str());
    fprintf(cppfp, "#define NCNN_INCLUDE_GUARD_%s\n", include_guard_var.c_str());

    fprintf(cppfp, "\n#ifdef _MSC_VER\n__declspec(align(4))\n#else\n__attribute__((aligned(4)))\n#endif\n");
    fprintf(cppfp, "static const unsigned char %s[] = {\n", param_var.c_str());

    int i = 0;
    while (!feof(mp))
    {
        int c = fgetc(mp);
        if (c == EOF)
            break;
        fprintf(cppfp, "0x%02x,", c);

        i++;
        if (i % 16 == 0)
        {
            fprintf(cppfp, "\n");
        }
    }

    fprintf(cppfp, "};\n");

    fclose(mp);

    // dump model
    std::string model_var = path_to_varname(modelpath);

    FILE* bp = fopen(modelpath, "rb");

    if (!bp)
    {
        fprintf(stderr, "fopen %s failed\n", modelpath);
        return -1;
    }

    fprintf(cppfp, "\n#ifdef _MSC_VER\n__declspec(align(4))\n#else\n__attribute__((aligned(4)))\n#endif\n");
    fprintf(cppfp, "static const unsigned char %s[] = {\n", model_var.c_str());

    i = 0;
    while (!feof(bp))
    {
        int c = fgetc(bp);
        if (c == EOF)
            break;
        fprintf(cppfp, "0x%02x,", c);

        i++;
        if (i % 16 == 0)
        {
            fprintf(cppfp, "\n");
        }
    }

    fprintf(cppfp, "};\n");

    fprintf(cppfp, "#endif // NCNN_INCLUDE_GUARD_%s\n", include_guard_var.c_str());

    fclose(bp);

    fclose(cppfp);

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s [ncnnproto] [ncnnbin] [idcpppath] [memcpppath]\n", argv[0]);
        return -1;
    }

    const char* parampath = argv[1];
    const char* modelpath = argv[2];
    const char* idcpppath = argv[3];
    const char* memcpppath = argv[4];

    std::string parambinpath = std::string(parampath) + ".bin";

    dump_param(parampath, parambinpath.c_str(), idcpppath);

    write_memcpp(parambinpath.c_str(), modelpath, memcpppath);

    return 0;
}
