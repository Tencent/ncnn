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
#include <stdio.h>
#include <string.h>
#include <cstddef>
#include <string>
#include <vector>
#include "layer.h"

static std::vector<std::string> layer_names;
static std::vector<std::string> blob_names;

static int find_blob_index_by_name(const char* name)
{
    for (std::size_t i=0; i<blob_names.size(); i++)
    {
        if (blob_names[i] == name)
        {
            return i;
        }
    }

    fprintf(stderr, "find_blob_index_by_name %s failed\n", name);
    return -1;
}

static void sanitize_name(char* name)
{
    for (std::size_t i=0; i<strlen(name); i++)
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

static int dump_param(const char* parampath, const char* parambinpath, const char* idcpppath)
{
    FILE* fp = fopen(parampath, "rb");

    FILE* mp = fopen(parambinpath, "wb");
    FILE* ip = fopen(idcpppath, "wb");

    std::string param_var = path_to_varname(parampath);

    std::string include_guard_var = path_to_varname(idcpppath);

    fprintf(ip, "#ifndef NCNN_INCLUDE_GUARD_%s\n", include_guard_var.c_str());
    fprintf(ip, "#define NCNN_INCLUDE_GUARD_%s\n", include_guard_var.c_str());
    fprintf(ip, "namespace %s_id {\n", param_var.c_str());

    int layer_count = 0;
    int blob_count = 0;
    fscanf(fp, "%d %d", &layer_count, &blob_count);
    fwrite(&layer_count, sizeof(int), 1, mp);
    fwrite(&blob_count, sizeof(int), 1, mp);

    layer_names.resize(layer_count);
    blob_names.resize(blob_count);

    int layer_index = 0;
    int blob_index = 0;
    while (!feof(fp))
    {
        int nscan = 0;

        char layer_type[32];
        char layer_name[256];
        int bottom_count = 0;
        int top_count = 0;
        nscan = fscanf(fp, "%32s %256s %d %d", layer_type, layer_name, &bottom_count, &top_count);
        if (nscan != 4)
        {
            continue;
        }

        sanitize_name(layer_name);

        int typeindex = ncnn::layer_to_index(layer_type);
        fwrite(&typeindex, sizeof(int), 1, mp);

        fwrite(&bottom_count, sizeof(int), 1, mp);
        fwrite(&top_count, sizeof(int), 1, mp);

        fprintf(ip, "const int LAYER_%s = %d;\n", layer_name, layer_index);

//         layer->bottoms.resize(bottom_count);
        for (int i=0; i<bottom_count; i++)
        {
            char bottom_name[256];
            nscan = fscanf(fp, "%256s", bottom_name);
            if (nscan != 1)
            {
                continue;
            }

            sanitize_name(bottom_name);

            int bottom_blob_index = find_blob_index_by_name(bottom_name);

            fwrite(&bottom_blob_index, sizeof(int), 1, mp);
        }

//         layer->tops.resize(top_count);
        for (int i=0; i<top_count; i++)
        {
            char blob_name[256];
            nscan = fscanf(fp, "%256s", blob_name);
            if (nscan != 1)
            {
                continue;
            }

            sanitize_name(blob_name);

            blob_names[blob_index] = std::string(blob_name);

            fprintf(ip, "const int BLOB_%s = %d;\n", blob_name, blob_index);

            fwrite(&blob_index, sizeof(int), 1, mp);

            blob_index++;
        }

        // dump layer specific params
        char buffer[1024];
        fgets(buffer, 1024, fp);

        int pos = 0;
        int nconsumed = 0;
        while (1)
        {
            // skip whitespace
            nconsumed = 0;
            sscanf(buffer + pos, "%*[ \t]%n", &nconsumed);
            pos += nconsumed;

            bool isfloat = false;
            // look ahead for determine isfloat
            const char* bp = buffer + pos;
            for (int j=0; j<20; j++)
            {
                if (bp[j] == ' ' || bp[j] == '\t')
                {
                    break;
                }
                if (bp[j] == '.')
                {
                    isfloat = true;
                    break;
                }
            }

            if (isfloat)
            {
                float vf;
                nconsumed = 0;
                nscan = sscanf(buffer + pos, "%f%n", &vf, &nconsumed);

                pos += nconsumed;

                if (nscan != 1)
                {
                    break;
                }

                fwrite(&vf, sizeof(float), 1, mp);
            }
            else
            {
                int v;
                nconsumed = 0;
                nscan = sscanf(buffer + pos, "%d%n", &v, &nconsumed);

                pos += nconsumed;

                if (nscan != 1)
                {
                    break;
                }

                fwrite(&v, sizeof(int), 1, mp);
            }

        }

        layer_names[layer_index] = std::string(layer_name);

        layer_index++;
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

    fprintf(cppfp, "#ifndef NCNN_INCLUDE_GUARD_%s\n", include_guard_var.c_str());
    fprintf(cppfp, "#define NCNN_INCLUDE_GUARD_%s\n", include_guard_var.c_str());

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

    const char* lastslash = strrchr(parampath, '/');
    const char* name = lastslash == NULL ? parampath : lastslash + 1;

    std::string parambinpath = std::string(name) + ".bin";

    dump_param(parampath, parambinpath.c_str(), idcpppath);

    write_memcpp(parambinpath.c_str(), modelpath, memcpppath);

    return 0;
}
