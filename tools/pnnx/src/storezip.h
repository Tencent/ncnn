// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef PNNX_STOREZIP_H
#define PNNX_STOREZIP_H

#include <map>
#include <string>
#include <vector>

namespace pnnx {

class StoreZipReader
{
public:
    StoreZipReader();
    ~StoreZipReader();

    int open(const std::string& path);

    size_t get_file_size(const std::string& name);

    int read_file(const std::string& name, char* data);

    int close();

private:
    FILE* fp;

    struct StoreZipMeta
    {
        size_t offset;
        size_t size;
    };

    std::map<std::string, StoreZipMeta> filemetas;
};

class StoreZipWriter
{
public:
    StoreZipWriter();
    ~StoreZipWriter();

    int open(const std::string& path);

    int write_file(const std::string& name, const char* data, size_t size);

    int close();

private:
    FILE* fp;

    struct StoreZipMeta
    {
        std::string name;
        size_t lfh_offset;
        uint32_t crc32;
        uint32_t size;
    };

    std::vector<StoreZipMeta> filemetas;
};

} // namespace pnnx

#endif // PNNX_STOREZIP_H
