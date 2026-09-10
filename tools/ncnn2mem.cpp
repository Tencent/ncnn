// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "datareader.h"
#include "layer.h"
#include "layer_type.h"
#include "paramdict.h"

#include <cstddef>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

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
    for (char* p = name; *p; p++)
    {
        if (!isalnum((unsigned char)*p))
        {
            *p = '_';
        }
    }
}

static std::string path_to_varname(const char* path)
{
    const char* lastslash = strrchr(path, '/');
    const char* name = lastslash == NULL ? path : lastslash + 1;

    std::string varname = name;
    for (std::size_t i = 0; i < varname.size(); i++)
    {
        if (!isalnum((unsigned char)varname[i]))
        {
            varname[i] = '_';
        }
    }

    return varname;
}

static bool write_param(FILE* fp, const void* data, size_t size)
{
    if (fwrite(data, 1, size, fp) != size)
    {
        fprintf(stderr, "write param failed\n");
        return false;
    }
    return true;
}

static int close_file(FILE* fp, const char* path)
{
    // fclose may report a buffered write failure even when earlier writes succeeded
    const int io_error = ferror(fp);
    const int close_error = fclose(fp);
    if (io_error || close_error != 0)
    {
        fprintf(stderr, "file io failed %s\n", path);
        return -1;
    }
    return 0;
}

class ParamDictText : public ncnn::ParamDict
{
public:
    using ncnn::ParamDict::load_param;
};

static int dump_param_values(FILE* fp, FILE* mp)
{
    ncnn::DataReaderFromStdio dr(fp);
    ParamDictText pd;
    if (pd.load_param(dr) != 0)
        return -1;
    if (ferror(fp))
    {
        fprintf(stderr, "read param failed\n");
        return -1;
    }

    for (int id = 0; id < NCNN_MAX_PARAM_COUNT; id++)
    {
        const int type = pd.type(id);
        if (type == 0)
            continue;
        if (type == 2)
        {
            const int value = pd.get(id, 0);
            if (!write_param(mp, &id, sizeof(int)) || !write_param(mp, &value, sizeof(int)))
                return -1;
        }
        else if (type == 3)
        {
            const float value = pd.get(id, 0.f);
            if (!write_param(mp, &id, sizeof(int)) || !write_param(mp, &value, sizeof(float)))
                return -1;
        }
        else if (type == 4 || type == 5 || type == 6)
        {
            const ncnn::Mat value = pd.get(id, ncnn::Mat());
            const int encoded_id = -id - 23300;
            const int len = value.w;
            if (!write_param(mp, &encoded_id, sizeof(int)) || !write_param(mp, &len, sizeof(int)))
                return -1;
            if (len > 0 && !write_param(mp, value.data, (size_t)len * sizeof(float)))
                return -1;
        }
        else if (type == 7)
        {
            const std::string value = pd.get(id, std::string());
            const int encoded_id = -id - 23400;
            const int len = (int)value.size();
            const char padding[3] = {0};
            const int padding_size = (4 - len % 4) % 4;
            if (!write_param(mp, &encoded_id, sizeof(int)) || !write_param(mp, &len, sizeof(int))
                    || !write_param(mp, value.data(), len) || !write_param(mp, padding, padding_size))
                return -1;
        }
        else
        {
            fprintf(stderr, "unsupported parameter type %d (id=%d)\n", type, id);
            return -1;
        }
    }

    const int eop = -233;
    return write_param(mp, &eop, sizeof(int)) ? 0 : -1;
}

static int dump_param_impl(FILE* fp, FILE* mp, FILE* ip, const char* parampath, const char* idcpppath)
{
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
    if (magic != 7767517)
    {
        fprintf(stderr, "param is too old, please regenerate\n");
        return -1;
    }
    if (!write_param(mp, &magic, sizeof(int)))
        return -1;

    int layer_count = 0;
    int blob_count = 0;
    nscan = fscanf(fp, "%d %d", &layer_count, &blob_count);
    if (nscan != 2)
    {
        fprintf(stderr, "read layer_count and blob_count failed %d\n", nscan);
        return -1;
    }
    if (layer_count <= 0 || blob_count <= 0)
    {
        fprintf(stderr, "invalid layer_count or blob_count\n");
        return -1;
    }
    if (!write_param(mp, &layer_count, sizeof(int)) || !write_param(mp, &blob_count, sizeof(int)))
        return -1;

    blob_names.resize(blob_count);

    std::vector<std::string> custom_layer_index;

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
        if (bottom_count < 0 || top_count < 0 || top_count > blob_count - blob_index)
        {
            fprintf(stderr, "invalid bottom_count or top_count (layer=%d)\n", i);
            return -1;
        }

        sanitize_name(layer_name);

        int typeindex = ncnn::layer_to_index(layer_type);
        if (typeindex == -1)
        {
            // lookup custom_layer_index
            for (size_t j = 0; j < custom_layer_index.size(); j++)
            {
                if (custom_layer_index[j] == layer_type)
                {
                    typeindex = ncnn::LayerType::CustomBit | j;
                    break;
                }
            }

            if (typeindex == -1)
            {
                // new custom layer type
                size_t j = custom_layer_index.size();
                custom_layer_index.push_back(layer_type);
                typeindex = ncnn::LayerType::CustomBit | j;
            }
        }
        if (!write_param(mp, &typeindex, sizeof(int))
                || !write_param(mp, &bottom_count, sizeof(int)) || !write_param(mp, &top_count, sizeof(int)))
            return -1;

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
            if (bottom_blob_index < 0)
                return -1;

            if (!write_param(mp, &bottom_blob_index, sizeof(int)))
                return -1;
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

            if (!write_param(mp, &blob_index, sizeof(int)))
                return -1;

            blob_index++;
        }

        if (dump_param_values(fp, mp) != 0)
            return -1;
        if (ferror(ip))
        {
            fprintf(stderr, "write %s failed\n", idcpppath);
            return -1;
        }
    }

    // dump custom layer index
    for (size_t j = 0; j < custom_layer_index.size(); j++)
    {
        const std::string& layer_type = custom_layer_index[j];
        int typeindex = ncnn::LayerType::CustomBit | j;

        fprintf(ip, "const int TYPEINDEX_%s = %d;\n", layer_type.c_str(), typeindex);

        fprintf(stderr, "net.register_custom_layer(%s_id::TYPEINDEX_%s, %s_layer_creator);\n", param_var.c_str(), layer_type.c_str(), layer_type.c_str());
    }

    fprintf(ip, "} // namespace %s_id\n", param_var.c_str());
    fprintf(ip, "#endif // NCNN_INCLUDE_GUARD_%s\n", include_guard_var.c_str());

    return 0;
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
    if (!mp)
    {
        fprintf(stderr, "fopen %s failed\n", parambinpath);
        close_file(fp, parampath);
        return -1;
    }

    FILE* ip = fopen(idcpppath, "wb");
    if (!ip)
    {
        fprintf(stderr, "fopen %s failed\n", idcpppath);
        close_file(fp, parampath);
        close_file(mp, parambinpath);
        return -1;
    }

    int ret = dump_param_impl(fp, mp, ip, parampath, idcpppath);
    if (close_file(fp, parampath) != 0)
        ret = -1;
    if (close_file(mp, parambinpath) != 0)
        ret = -1;
    if (close_file(ip, idcpppath) != 0)
        ret = -1;
    return ret;
}

static int write_memcpp_impl(FILE* mp, FILE* bp, FILE* cppfp, const char* parambinpath, const char* modelpath, const char* memcpppath)
{
    // dump param
    std::string param_var = path_to_varname(parambinpath);

    std::string include_guard_var = path_to_varname(memcpppath);

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
        if (fprintf(cppfp, "0x%02x,", c) < 0)
        {
            fprintf(stderr, "write %s failed\n", memcpppath);
            return -1;
        }

        i++;
        if (i % 16 == 0)
        {
            fprintf(cppfp, "\n");
        }
    }

    fprintf(cppfp, "};\n");

    if (ferror(mp))
    {
        fprintf(stderr, "read %s failed\n", parambinpath);
        return -1;
    }

    // dump model
    std::string model_var = path_to_varname(modelpath);

    fprintf(cppfp, "\n#ifdef _MSC_VER\n__declspec(align(4))\n#else\n__attribute__((aligned(4)))\n#endif\n");
    fprintf(cppfp, "static const unsigned char %s[] = {\n", model_var.c_str());

    i = 0;
    while (!feof(bp))
    {
        int c = fgetc(bp);
        if (c == EOF)
            break;
        if (fprintf(cppfp, "0x%02x,", c) < 0)
        {
            fprintf(stderr, "write %s failed\n", memcpppath);
            return -1;
        }

        i++;
        if (i % 16 == 0)
        {
            fprintf(cppfp, "\n");
        }
    }

    fprintf(cppfp, "};\n");

    fprintf(cppfp, "#endif // NCNN_INCLUDE_GUARD_%s\n", include_guard_var.c_str());

    return 0;
}

static int write_memcpp(const char* parambinpath, const char* modelpath, const char* memcpppath)
{
    FILE* mp = fopen(parambinpath, "rb");
    if (!mp)
    {
        fprintf(stderr, "fopen %s failed\n", parambinpath);
        return -1;
    }

    FILE* bp = fopen(modelpath, "rb");
    if (!bp)
    {
        fprintf(stderr, "fopen %s failed\n", modelpath);
        close_file(mp, parambinpath);
        return -1;
    }

    FILE* cppfp = fopen(memcpppath, "wb");
    if (!cppfp)
    {
        fprintf(stderr, "fopen %s failed\n", memcpppath);
        close_file(mp, parambinpath);
        close_file(bp, modelpath);
        return -1;
    }

    int ret = write_memcpp_impl(mp, bp, cppfp, parambinpath, modelpath, memcpppath);
    if (close_file(mp, parambinpath) != 0)
        ret = -1;
    if (close_file(bp, modelpath) != 0)
        ret = -1;
    if (close_file(cppfp, memcpppath) != 0)
        ret = -1;
    return ret;
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

    if (dump_param(parampath, parambinpath.c_str(), idcpppath) != 0)
        return -1;

    return write_memcpp(parambinpath.c_str(), modelpath, memcpppath);
}
