// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#include <stdio.h>
#include <string.h>
#include <string>

static int copy_param(const char* parampath, FILE* outparamfp, int* total_layer_count, int* total_blob_count)
{
    // resolve model namespace from XYZ.param
    const char* lastslash = strrchr(parampath, '/');
    const char* name = lastslash == NULL ? parampath : lastslash + 1;
    const char* dot = strrchr(name, '.');
    std::string ns = dot ? std::string(name).substr(0, dot - name) : std::string(name);

    FILE* fp = fopen(parampath, "rb");

    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", parampath);
        return -1;
    }

    int nscan = 0;
    int magic = 0;
    nscan = fscanf(fp, "%d", &magic);
    if (nscan != 1 || magic != 7767517)
    {
        fprintf(stderr, "read magic failed %d\n", nscan);
        return -1;
    }

    int layer_count = 0;
    int blob_count = 0;
    nscan = fscanf(fp, "%d %d", &layer_count, &blob_count);
    if (nscan != 2)
    {
        fprintf(stderr, "read layer_count and blob_count failed %d\n", nscan);
        return -1;
    }

    *total_layer_count += layer_count;
    *total_blob_count += blob_count;

    char line[1024];
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

        fprintf(outparamfp, "%-24s %s/%-24s %d %d", layer_type, ns.c_str(), layer_name, bottom_count, top_count);

        for (int j = 0; j < bottom_count; j++)
        {
            char bottom_name[257];
            nscan = fscanf(fp, "%256s", bottom_name);
            if (nscan != 1)
            {
                fprintf(stderr, "read bottom_name failed %d\n", nscan);
                return -1;
            }

            fprintf(outparamfp, " %s/%s", ns.c_str(), bottom_name);
        }

        for (int j = 0; j < top_count; j++)
        {
            char top_name[257];
            nscan = fscanf(fp, "%256s", top_name);
            if (nscan != 1)
            {
                fprintf(stderr, "read top_name failed %d\n", nscan);
                return -1;
            }

            fprintf(outparamfp, " %s/%s", ns.c_str(), top_name);
        }

        // copy param dict string
        char* s = fgets(line, 1024, fp);
        if (!s)
        {
            fprintf(stderr, "read line %s failed\n", parampath);
            break;
        }

        fputs(line, outparamfp);
    }

    fclose(fp);

    return 0;
}

static int copy_bin(const char* binpath, FILE* outbinfp)
{
    FILE* fp = fopen(binpath, "rb");

    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", binpath);
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    int len = (int)ftell(fp);
    rewind(fp);

    char buffer[4096];
    int i = 0;
    for (; i + 4095 < len;)
    {
        size_t nread = fread(buffer, 1, 4096, fp);
        size_t nwrite = fwrite(buffer, 1, nread, outbinfp);
        i += nwrite;
    }
    {
        size_t nread = fread(buffer, 1, len - i, fp);
        size_t nwrite = fwrite(buffer, 1, nread, outbinfp);
        i += nwrite;
    }

    if (i != len)
    {
        fprintf(stderr, "copy %s incomplete\n", binpath);
    }

    fclose(fp);

    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 7 || (argc - 1) % 2 != 0)
    {
        fprintf(stderr, "Usage: %s [param1] [bin1] [param2] [bin2] ... [outparam] [outbin]\n", argv[0]);
        return -1;
    }

    const char* outparampath = argv[argc - 2];
    const char* outbinpath = argv[argc - 1];

    FILE* outparamfp = fopen(outparampath, "wb");
    FILE* outbinfp = fopen(outbinpath, "wb");

    // magic
    fprintf(outparamfp, "7767517\n");

    // layer count and blob count placeholder
    // 99999 is large enough I think  --- nihui
    fprintf(outparamfp, "           \n");

    int total_layer_count = 0;
    int total_blob_count = 0;

    const int model_count = (argc - 3) / 2;

    for (int i = 0; i < model_count; i++)
    {
        const char* parampath = argv[i * 2 + 1];
        const char* binpath = argv[i * 2 + 2];

        copy_param(parampath, outparamfp, &total_layer_count, &total_blob_count);
        copy_bin(binpath, outbinfp);
    }

    // the real layer count and blob count
    rewind(outparamfp);
    fprintf(outparamfp, "7767517\n");
    fprintf(outparamfp, "%d %d", total_layer_count, total_blob_count);

    fclose(outparamfp);
    fclose(outbinfp);

    return 0;
}
