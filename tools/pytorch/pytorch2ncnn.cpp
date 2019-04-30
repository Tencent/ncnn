// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <map>
#include <set>
#include <string>
#include <vector>

struct zip_local_file_header_t
{
    uint32_t signature;
    uint16_t version;
    uint16_t flag;
    uint16_t compression_method;
    uint16_t last_modify_time;
    uint16_t last_modify_data;
    uint32_t crc32;
    uint32_t compressed_size;
    uint32_t uncompressed_size;
    uint16_t filename_length;
    uint16_t extradata_length;
} __attribute__((packed));

union zip_data_descriptor_t
{
    struct
    {
        uint32_t signature;
        uint32_t crc32;
        uint32_t compressed_size;
        uint32_t uncompressed_size;
    } ws;
    struct
    {
        uint32_t crc32;
        uint32_t compressed_size;
        uint32_t uncompressed_size;
    } os;
} __attribute__((packed));

struct zip_central_directory_file_header_t
{
    uint32_t signature;
    uint16_t version_made_by;
    uint16_t version;
    uint16_t flag;
    uint16_t compression_method;
    uint16_t last_modify_time;
    uint16_t last_modify_data;
    uint32_t crc32;
    uint32_t compressed_size;
    uint32_t uncompressed_size;
    uint16_t filename_length;
    uint16_t extradata_length;
    uint16_t filecomment_length;
    uint16_t disk_number;
    uint16_t internal_file_attrs;
    uint32_t external_file_attrs;
    uint32_t relative_local_file_header_offset;
} __attribute__((packed));

struct zip_end_of_central_directory_record_t
{
    uint32_t signature;
    uint16_t disk_number;
    uint16_t disk_starts;
    uint16_t disk_record_count;
    uint16_t total_record_count;
    uint32_t central_directory_size;
    uint32_t central_directory_offset;
    uint16_t comment_length;
} __attribute__((packed));

struct zip64_end_of_central_directory_record_t
{
    uint32_t signature;
    uint64_t header_size;
    uint16_t version_made_by;
    uint16_t version;
    uint32_t disk_number;
    uint32_t disk_number_with_central_directory;
    uint64_t total_entry_number_on_disk;
    uint64_t total_entry_number;
    uint64_t central_directory_size;
    uint64_t central_directory_offset;
} __attribute__((packed));

struct zip64_end_of_central_directory_locator_t
{
    uint32_t signature;
    uint32_t disk_number_with_central_directory;
    uint64_t end_central_directory_offset;
    uint32_t total_disk_number;
} __attribute__((packed));

static uint32_t CRC32_TABLE[256];

static void CRC32_TABLE_INIT()
{
    for (int i=0; i<256; i++)
    {
        uint32_t c = i;
        for (int j=0; j<8; j++)
        {
            if (c & 1)
                c = (c >> 1) ^ 0xedb88320;
            else
                c >>= 1;
        }
        CRC32_TABLE[i] = c;
    }
}

static uint32_t CRC32(uint32_t x, unsigned char ch)
{
    return (x >> 8) ^ CRC32_TABLE[(x ^ ch) & 0xff];
}

static uint32_t CRC32_buffer(const unsigned char* data, int len)
{
    uint32_t x = 0xffffffff;

    for (int i=0; i<len; i++)
        x = CRC32(x, data[i]);

    return x ^ 0xffffffff;
}

static int read_zip_local_file(std::string& filename, std::string& filedata, FILE* fp)
{
    filename.clear();
    filedata.clear();

    zip_local_file_header_t h;
    fread(&h, sizeof(h), 1, fp);

    filename.resize(h.filename_length);
    fread((void*)filename.data(), 1, h.filename_length, fp);

    // skip extradata
    fseek(fp, h.extradata_length, SEEK_CUR);

    if (h.flag & 0x08)
    {
        uint32_t x = 0xffffffff;

        for (;;)
        {
            unsigned char ch = (unsigned char)fgetc(fp);

            zip_data_descriptor_t d;
            fread(&d, sizeof(d), 1, fp);

            filedata.push_back(ch);

            x = CRC32(x, ch);

            if (filedata.size() == d.ws.compressed_size || filedata.size() == d.os.compressed_size)
            {
                uint32_t crc32 = x ^ 0xffffffff;

                if (d.ws.signature == 0x08074b50 && crc32 == d.ws.crc32)
                    break;

                if (crc32 == d.os.crc32)
                {
                    fseek(fp, -sizeof(d.ws.signature), SEEK_CUR);
                    break;
                }
            }

            fseek(fp, -sizeof(d), SEEK_CUR);
        }
    }
    else
    {
        // use h.compressed_size
        // use h.uncompressed_size
        // use h.crc32
        filedata.resize(h.compressed_size);
        fread((void*)filedata.data(), 1, h.compressed_size, fp);

        uint32_t crc32 = CRC32_buffer((const unsigned char*)filedata.data(), filedata.size());
        if (crc32 != h.crc32)
        {
            fprintf(stderr, "crc32 mismatch\n");
            return -1;
        }
    }

    return 0;
}

static int read_zip_central_directory_file(std::string& filename, FILE* fp)
{
    filename.clear();

    zip_central_directory_file_header_t h;
    fread(&h, sizeof(h), 1, fp);

    filename.resize(h.filename_length);
    fread((void*)filename.data(), 1, h.filename_length, fp);

    // skip extradata
    fseek(fp, h.extradata_length, SEEK_CUR);

    // skip filecomment
    fseek(fp, h.filecomment_length, SEEK_CUR);

    return 0;
}

static int read_zip_end_of_central_directory_record(FILE* fp)
{
    zip_end_of_central_directory_record_t h;
    fread(&h, sizeof(h), 1, fp);

    // skip comment
    fseek(fp, h.comment_length, SEEK_CUR);

    return 0;
}

static int read_zip64_end_of_central_directory_record(FILE* fp)
{
    zip64_end_of_central_directory_record_t h;
    fread(&h, sizeof(h), 1, fp);

    return 0;
}

static int read_zip64_end_of_central_directory_locator(FILE* fp)
{
    zip64_end_of_central_directory_locator_t h;
    fread(&h, sizeof(h), 1, fp);

    return 0;
}

static int read_pt(const char* ptpath, std::string& code, std::string& model_json, std::map<int, std::string>& tensors)
{
    CRC32_TABLE_INIT();

    // filename
    const char* last_slash_pos = strrchr(ptpath, '/');
    const char* ptname = last_slash_pos ? last_slash_pos + 1 : ptpath;

    // extension
    const char* extname = strchr(ptname, '.');
    std::string name = ptname;
    if (extname)
        name = name.substr(0, extname - ptname);

    fprintf(stderr, "name = %s\n", name.c_str());

    FILE* fp = fopen(ptpath, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", ptpath);
        return -1;
    }

    while (!feof(fp))
    {
        // peek signature
        uint32_t signature;
        int nread = fread(&signature, sizeof(signature), 1, fp);
        if (nread != 1)
            break;
        fseek(fp, -sizeof(signature), SEEK_CUR);

        if (signature == 0x04034b50)
        {
            std::string filename;
            std::string filedata;
            read_zip_local_file(filename, filedata, fp);

            fprintf(stderr, "file %s = %lu\n", filename.c_str(), filedata.size());

            if (filename == name + "/code/" + name + ".py")
            {
                code = filedata;
            }
            else if (filename == name + "/model.json")
            {
                model_json = filedata;
            }
            else
            {
                int tensorId;
                std::string templ = name + "/tensors/%d";
                int nscan = sscanf(filename.c_str(), templ.c_str(), &tensorId);
                if (nscan == 1)
                {
                    tensors.insert(std::make_pair(tensorId, filedata));
                }
            }
        }
        else if (signature == 0x02014b50)
        {
            std::string filename;
            read_zip_central_directory_file(filename, fp);

            fprintf(stderr, "dirfile = %s\n", filename.c_str());
        }
        else if (signature == 0x06054b50)
        {
            read_zip_end_of_central_directory_record(fp);
        }
        else if (signature == 0x06064b50)
        {
            read_zip64_end_of_central_directory_record(fp);
        }
        else if (signature == 0x07064b50)
        {
            read_zip64_end_of_central_directory_locator(fp);
        }
        else
        {
            fprintf(stderr, "unhandled %x\n", signature);
            break;
        }
    }

    fclose(fp);

    return 0;
}

static const char* last_strstr(const char* haystack, const char* needle)
{
    if (*needle == '\0')
        return haystack;

    const char* result = 0;
    for (;;)
    {
        const char* p = strstr(haystack, needle);
        if (!p)
            break;

        result = p;
        haystack = p + 1;
    }

    return result;
}

class TensorInfo
{
public:
    int id;
    std::string name;
    std::string submodule_name;
    std::vector<int> dims;
    std::vector<int> strides;
};

static int parse_submodule_parameter_dict(const char* paramdict, int& _tensorId, std::string& _name)
{
    // "isBuffer":false,"tensorId":"0","name":"weight"
    {
        const char* keyvalue = strstr(paramdict, "\"tensorId\":");
        if (!keyvalue)
        {
            fprintf(stderr, "no tensorId\n");
            return -1;
        }

        int tensorId;
        int nscan = sscanf(keyvalue, "\"tensorId\":\"%d\"", &tensorId);
        if (nscan != 1)
        {
            fprintf(stderr, "no tensorId\n");
            return -1;
        }

        _tensorId = tensorId;
    }

    {
        const char* keyvalue = strstr(paramdict, "\"name\":");
        if (!keyvalue)
        {
            fprintf(stderr, "no name\n");
            return -1;
        }

        char name[256];
        int nscan = sscanf(keyvalue, "\"name\":\"%255[^\"]\"", name);
        if (nscan != 1)
        {
            fprintf(stderr, "no tensorId\n");
            return -1;
        }

        _name = name;
    }

    return 0;
}

static void parse_submodule_parameter_list(const char* s, std::vector<int>& tensorIds, std::vector<std::string>& names)
{
// {"isBuffer":false,"tensorId":"0","name":"weight"},{"isBuffer":false,"tensorId":"1","name":"bias"}

    tensorIds.clear();
    names.clear();

    const char* ps = s + 1;// +1 to skip leading "{"

    const char* parameter_end = strstr(ps, "},{");
    while (parameter_end)
    {
        std::string paramdict_string(ps, parameter_end - ps);

//         fprintf(stderr, "paramdict_string = %s\n", paramdict_string.c_str());

        int tensorId;
        std::string name;
        int pr = parse_submodule_parameter_dict(paramdict_string.c_str(), tensorId, name);
        if (pr == 0)
        {
            tensorIds.push_back(tensorId);
            names.push_back(name);
        }

        ps = parameter_end + 3;// +3 to skip "},{"
        parameter_end = strstr(ps, "},{");
    }

    std::string paramdict_string(ps, strlen(ps) - 1);

//     fprintf(stderr, "paramdict_string = %s\n", paramdict_string.c_str());

    int tensorId;
    std::string name;
    int pr = parse_submodule_parameter_dict(paramdict_string.c_str(), tensorId, name);
    if (pr == 0)
    {
        tensorIds.push_back(tensorId);
        names.push_back(name);
    }
}

static std::vector<int> parse_int_array(const char* s)
{
    // "512","512","3","3","3"

    std::vector<int> array;

    char* ps = (char*)s;
    for (char* is = strtok(ps, ","); is; is = strtok(NULL, ","))
    {
        int i;
        int nscan = sscanf(is, "\"%d\"", &i);
        if (nscan != 1)
        {
            fprintf(stderr, "invalid int element\n");
            continue;
        }

        array.push_back(i);
    }

    return array;
}

static int parse_tensor_info_dict(const char* paramdict, int& _tensorId, std::vector<int>& _dims, std::vector<int>& _strides)
{
    // {"dims":["512","512","3","3","3"],"offset":"0","strides":["13824","27","9","3","1"],"requiresGrad":true,"dataType":"FLOAT","data":{"key":"tensors/0"},"device":"cpu"}
    {
        const char* keyvalue = strstr(paramdict, "\"key\":\"tensors/");
        if (!keyvalue)
        {
            fprintf(stderr, "no tensor data key\n");
            return -1;
        }

        int tensorId;
        int nscan = sscanf(keyvalue, "\"key\":\"tensors/%d\"", &tensorId);
        if (nscan != 1)
        {
            fprintf(stderr, "no tensorId\n");
            return -1;
        }

        _tensorId = tensorId;
    }

    {
        const char* keyvalue = strstr(paramdict, "\"dims\":");
        if (!keyvalue)
        {
            fprintf(stderr, "no tensor dims\n");
            return -1;
        }

        char dims[256];
        int nscan = sscanf(keyvalue, "\"dims\":[%255[^]]]", dims);
        if (nscan != 1)
        {
            fprintf(stderr, "no dims\n");
            return -1;
        }

        _dims = parse_int_array(dims);
    }

    {
        const char* keyvalue = strstr(paramdict, "\"strides\":");
        if (!keyvalue)
        {
            fprintf(stderr, "no tensor strides\n");
            return -1;
        }

        char strides[256];
        int nscan = sscanf(keyvalue, "\"strides\":[%255[^]]]", strides);
        if (nscan != 1)
        {
            fprintf(stderr, "no strides\n");
            return -1;
        }

        _strides = parse_int_array(strides);
    }

    return 0;
}

static int read_model_json(const std::string& model_json, std::vector<TensorInfo>& tensorinfos)
{
    tensorinfos.clear();

    const char* ps = model_json.c_str();

    const char* submodules = last_strstr(ps, "{\"submodules\":[");
    if (!submodules)
    {
        fprintf(stderr, "no submodules\n");
        return -1;
    }

    ps = submodules + sizeof("{\"submodules\":[") - 1;

    for (;;)
    {
        const char* parameters = strstr(ps, "{\"parameters\":[");
        if (!parameters)
            break;

        const char* parameters_end = strstr(parameters, "],");
        if (!parameters_end)
            break;

        parameters += sizeof("{\"parameters\":[") - 1;

        std::string parameters_string(parameters, parameters_end - parameters);

//         fprintf(stderr, "parameters = %s\n", parameters_string.c_str());

        ps = parameters_end + sizeof("],") - 1;

        const char* name = strstr(ps, "\"name\":\"");
        if (!name)
            break;

        const char* name_end = strstr(name, "\",");
        if (!name_end)
            break;

        name += sizeof("\"name\":\"") - 1;

        std::string name_string(name, name_end - name);

//         fprintf(stderr, "name = %s\n", name_string.c_str());

        ps = name_end + sizeof("\",") - 1;

        std::vector<int> tensorIds;
        std::vector<std::string> names;
        parse_submodule_parameter_list(parameters_string.c_str(), tensorIds, names);

        for (int i=0; i<(int)tensorIds.size(); i++)
        {
            TensorInfo ti;
            ti.id = tensorIds[i];
            ti.name = names[i];
            ti.submodule_name = name_string;
            tensorinfos.push_back(ti);
        }
    }

    const char* tensors = strstr(ps, "\"tensors\":[");
    if (!tensors)
    {
        fprintf(stderr, "no tensors\n");
        return -1;
    }

    ps = tensors + sizeof("\"tensors\":[") - 1;

    for (;;)
    {
        const char* tensor_end = strstr(ps, "},{");
        if (!tensor_end)
        {
            tensor_end = strstr(ps, "}]");
            if (!tensor_end)
            {
                break;
            }
        }

        tensor_end += 1;// +1 to include "}"

        std::string tensor_string(ps, tensor_end - ps);

//         fprintf(stderr, "tensor = %s\n", tensor_string.c_str());

        ps = tensor_end + 1;// +1 to skip ","

        int tensorId = -1;
        std::vector<int> dims;
        std::vector<int> strides;
        parse_tensor_info_dict(tensor_string.c_str(), tensorId, dims, strides);

        for (int i=0; i<(int)tensorinfos.size(); i++)
        {
            if (tensorinfos[i].id != tensorId)
                continue;

            tensorinfos[i].dims = dims;
            tensorinfos[i].strides = strides;
        }
    }

    return 0;
}

class PyTorchNode
{
public:
    std::string op;
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::string> weights;
    std::vector<std::string> args;
    std::vector<int> weight_tensors;
};

static std::vector<std::string> parse_op_output_list(const char* s)
{
    std::vector<std::string> outputs;

    const char* ps = s;

    for (;;)
    {
        if (*ps == '\0')
        {
            break;
        }

        if (*ps == ',' || *ps == ' ')
        {
            ps++;// skip ", "
            continue;
        }

        const char* output_end = strchr(ps, ',');
        if (!output_end)
        {
            outputs.push_back(ps);
            break;
        }

        std::string a(ps, output_end - ps);
        outputs.push_back(a);

        ps = output_end + 1;
    }

    return outputs;
}

static std::vector<std::string> parse_op_arg_list(const char* s)
{
    std::vector<std::string> args;

    const char* ps = s;

    for (;;)
    {
        if (*ps == '\0')
        {
            break;
        }

        if (*ps == ',' || *ps == ' ')
        {
            ps++;// skip ", "
            continue;
        }

        if (*ps == '[')
        {
            const char* list_end = strchr(ps, ']');
            if (!list_end)
            {
                fprintf(stderr, "unterminatied arg list\n");
                break;
            }

            std::string a(ps, list_end - ps + 1);
            args.push_back(a);

            ps = list_end + 1;
        }
        else if (strncmp(ps, "getattr(", 8) == 0 || strncmp(ps, "torch.t(", 8) == 0)
        {
            const char* deco_end = strchr(ps, ')');
            if (!deco_end)
            {
                fprintf(stderr, "unterminatied arg list\n");
                break;
            }

            const char* arg_end = strchr(deco_end, ',');
            if (!arg_end)
            {
                args.push_back(ps);
                break;
            }

            std::string a(ps, arg_end - ps);
            args.push_back(a);

            ps = arg_end + 1;
        }
        else
        {
            const char* arg_end = strchr(ps, ',');
            if (!arg_end)
            {
                arg_end = strchr(ps, ')');
                if (!arg_end)
                {
                    args.push_back(ps);
                    break;
                }
            }

            std::string a(ps, arg_end - ps);
            args.push_back(a);

            ps = arg_end + 1;
        }
    }

    return args;
}

static int read_code(const std::string& code, std::vector<PyTorchNode>& nodes)
{
    nodes.clear();

    int internal_unknown = 0;

    // read code line
    bool forward_input = false;

    char* ps = (char*)code.c_str();
    for (char* line = strtok(ps, "\n"); line; line = strtok(NULL, "\n"))
    {
//         fprintf(stderr, "line = %s\n", line);

        if (strstr(line, "op_version_set = "))
        {
            int op_version_set = 0;
            int nscan = sscanf(line, "op_version_set = %d", &op_version_set);
            if (nscan != 1)
                continue;

            fprintf(stderr, "op_version_set = %d\n", op_version_set);
        }
        else if (strstr(line, "def forward("))
        {
            if (strcmp(line, "def forward(self,") == 0)
            {
                forward_input = true;
                continue;
            }
        }
        else if (strstr(line, " -> "))
        {
            char netinput[256];
            int nscan = sscanf(line, "    %255[^:]: Tensor)  %*[^:]:", netinput);
            if (nscan != 1)
                continue;

//             fprintf(stderr, "netinput = %s\n", netinput);
            forward_input = false;

            PyTorchNode n;
            n.op = "Input";
            n.outputs.push_back(netinput);

            {
                // assign default unknown name
                char unknownname[256];
                sprintf(unknownname, "unknownncnn_%d", internal_unknown);

                n.name = unknownname;

                internal_unknown++;
            }

            nodes.push_back(n);
        }
        else if (strstr(line, " = torch."))
        {
            const char* op_args_start = strchr(line, '(');
            const char* op_args_end = strrchr(line, ')');
            if (!op_args_start || !op_args_end)
            {
                fprintf(stderr, "no operator args\n");
                continue;
            }

            std::string op_args(op_args_start + 1, op_args_end - op_args_start - 1);

            // input_1 = torch._convolution(data, ...)
            char outputs[256];
            char op[256];
            int nscan = sscanf(line, "  %255[^=]= %255[^(]", outputs, op);
            if (nscan != 2)
                continue;

            outputs[strlen(outputs) - 1] = '\0';// remove tail space

//             fprintf(stderr, "op = %s\n", op);
//             fprintf(stderr, "outputs = %s\n", outputs);
            fprintf(stderr, "op_args = %s\n", op_args.c_str());

            PyTorchNode n;
            n.op = op;
            n.outputs = parse_op_output_list(outputs);
            n.args = parse_op_arg_list(op_args.c_str());

            {
                // assign default unknown name
                char unknownname[256];
                sprintf(unknownname, "unknownncnn_%d", internal_unknown);

                n.name = unknownname;

                internal_unknown++;
            }

            nodes.push_back(n);
        }
        else if (strstr(line, "return "))
        {
            const char* op_args_start = strchr(line, '(');
            const char* op_args_end = strrchr(line, ')');
            if (!op_args_start || !op_args_end)
            {
                fprintf(stderr, "no operator args\n");
                continue;
            }

            std::string op_args(op_args_start + 1, op_args_end - op_args_start - 1);

            // return torch.threshold_(input, ...)
            char op[256];
            int nscan = sscanf(line, "  return %255[^(]", op);
            if (nscan != 1)
                continue;

//             fprintf(stderr, "op = %s\n", op);
//             fprintf(stderr, "outputs = ncnnoutput_0\n");
            fprintf(stderr, "op_args = %s\n", op_args.c_str());

            PyTorchNode n;
            n.op = op;
            n.outputs.push_back("ncnnoutput_0");
            n.args = parse_op_arg_list(op_args.c_str());

            {
                // assign default unknown name
                char unknownname[256];
                sprintf(unknownname, "unknownncnn_%d", internal_unknown);

                n.name = unknownname;

                internal_unknown++;
            }

            nodes.push_back(n);
        }
    }

    return 0;
}

int main(int argc, char** argv)
{
//     const char* ptpath = "model_base.pt";
//     const char* ptpath = "model_first.pt";
    const char* ptpath = "model_second.pt";
    const char* ncnn_prototxt = "ncnn.param";
    const char* ncnn_modelbin = "ncnn.bin";

    std::string code;
    std::string model_json;
    std::map<int, std::string> tensors;
    read_pt(ptpath, code, model_json, tensors);

//     fprintf(stderr, "code = %s\n", code.c_str());
//     fprintf(stderr, "model_json = %s\n", model_json.c_str());
//     fprintf(stderr, "tensors = %lu\n", tensors.size());

    std::vector<TensorInfo> tensorinfos;
    read_model_json(model_json, tensorinfos);

    for (int i=0; i<(int)tensorinfos.size(); i++)
    {
        const TensorInfo& ti = tensorinfos[i];
        fprintf(stderr, "%d %s/%s ", ti.id, ti.submodule_name.c_str(), ti.name.c_str());

        fprintf(stderr, "[ ");
        for (int j=0; j<(int)ti.dims.size(); j++)
        {
            fprintf(stderr, "%d ", ti.dims[j]);
        }
        fprintf(stderr, "]");

        fprintf(stderr, "\n");
    }

    std::vector<PyTorchNode> nodes;
    read_code(code, nodes);

    FILE* pp = fopen(ncnn_prototxt, "wb");
    FILE* bp = fopen(ncnn_modelbin, "wb");

    // magic
    fprintf(pp, "7767517\n");

    const int node_count = nodes.size();

    fprintf(stderr, "node_count = %d\n", node_count);

    // node reference
    std::map<std::string, int> node_reference;

    // weight node
    std::vector<int> weight_nodes;

    // global definition line
    // [layer count] [blob count]
    std::set<std::string> blob_names;
    for (int i=0; i<node_count; i++)
    {
        PyTorchNode& n = nodes[i];

        for (int j=0; j<(int)n.outputs.size(); j++)
        {
            blob_names.insert(n.outputs[j]);
        }

        // distinguish weights and inputs
        std::vector<std::string> op_arg_list = n.args;
        std::vector<std::string> input_list;
        std::vector<std::string> weight_list;
        std::vector<std::string> arg_list;
        for (int i=0; i<(int)op_arg_list.size(); i++)
        {
            const std::string& arg = op_arg_list[i];
            if (blob_names.find(arg) == blob_names.end())
            {
                // self.xyz.submodule_name.name
                // getattr(self.xyz, "submodule_name").name
                // torch.t(self.xyz.submodule_name.name)
                const char* argstr = arg.c_str();
                if (strncmp(argstr, "self.", 5) == 0 || strncmp(argstr, "getattr(", 8) == 0 || strncmp(argstr, "torch.t(", 8) == 0)
                {
                    weight_list.push_back(arg);
                }
                else
                {
                    arg_list.push_back(arg);
                }
            }
            else
            {
                input_list.push_back(arg);

                if (node_reference.find(arg) == node_reference.end())
                {
                    node_reference[arg] = 1;
                }
                else
                {
                    node_reference[arg] = node_reference[arg] + 1;
                }
            }
        }

        n.inputs = input_list;
        n.weights = weight_list;
        n.args = arg_list;

        // TODO parse weights for weight_tensors
    }

    // remove node_reference entry with reference equals to one
    int splitncnn_blob_count = 0;
    std::map<std::string, int>::iterator it = node_reference.begin();
    while (it != node_reference.end())
    {
        if (it->second == 1)
        {
            node_reference.erase(it++);
        }
        else
        {
            splitncnn_blob_count += it->second;
//             fprintf(stderr, "%s %d\n", it->first.c_str(), it->second);
            ++it;
        }
    }

    fprintf(pp, "%lu %lu\n", node_count + node_reference.size(), blob_names.size() + splitncnn_blob_count);

    int internal_split = 0;

    for (int i=0; i<node_count; i++)
    {
        const PyTorchNode& n = nodes[i];

        if (n.op == "torch.addmm")
        {
            fprintf(pp, "%-16s", "InnerProduct");
        }
        else if (n.op == "torch._convolution")
        {
            fprintf(pp, "%-16s", "Convolution");
        }
        else if (n.op == "torch.dropout")
        {
            fprintf(pp, "%-16s", "Dropout");
        }
        else if (n.op == "torch.relu_")
        {
            fprintf(pp, "%-16s", "ReLU");
        }
        else if (n.op == "torch.softmax")
        {
            fprintf(pp, "%-16s", "Softmax");
        }
        else if (n.op == "torch.view")
        {
            fprintf(pp, "%-16s", "Reshape");
        }
        else
        {
            fprintf(pp, "%-16s", n.op.c_str());
        }

        fprintf(pp, " %-24s %d %d", n.name.c_str(), (int)n.inputs.size(), (int)n.outputs.size());

        for (int j=0; j<(int)n.inputs.size(); j++)
        {
            std::string input_name = n.inputs[j];

            if (node_reference.find(input_name) != node_reference.end())
            {
                int refidx = node_reference[input_name] - 1;
                node_reference[input_name] = refidx;

                char splitsuffix[256];
                sprintf(splitsuffix, "_splitncnn_%d", refidx);
                input_name = input_name + splitsuffix;
            }

            fprintf(pp, " %s", input_name.c_str());
        }

        for (int j=0; j<(int)n.outputs.size(); j++)
        {
            fprintf(pp, " %s", n.outputs[j].c_str());
        }

        // TODO op specific params
        {
            for (int j=0; j<(int)n.args.size(); j++)
            {
                fprintf(pp, " %s", n.args[j].c_str());
            }
        }

        fprintf(pp, "\n");

        for (int j=0; j<(int)n.outputs.size(); j++)
        {
            const std::string& output_name = n.outputs[j];
            if (node_reference.find(output_name) != node_reference.end())
            {
                int refcount = node_reference[output_name];
                if (refcount > 1)
                {
                    char splitname[256];
                    sprintf(splitname, "splitncnn_%d", internal_split);
                    fprintf(pp, "%-16s %-24s %d %d", "Split", splitname, 1, refcount);

                    fprintf(pp, " %s", output_name.c_str());

                    for (int k=0; k<refcount; k++)
                    {
                        fprintf(pp, " %s_splitncnn_%d", output_name.c_str(), k);
                    }
                    fprintf(pp, "\n");

                    internal_split++;
                }
            }
        }
    }

    fclose(pp);
    fclose(bp);

    return 0;
}
