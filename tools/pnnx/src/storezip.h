// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_STOREZIP_H
#define PNNX_STOREZIP_H

#include <stdio.h>
#include <stdint.h>
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

    std::vector<std::string> get_names() const;

    uint64_t get_file_size(const std::string& name) const;

    int read_file(const std::string& name, char* data);
    int read_file(const std::string& name, std::vector<unsigned char>& data);

    int close();

    std::string error;

private:
    struct StoreZipMeta
    {
        uint64_t offset;
        uint64_t size;
        uint32_t crc32;
        uint16_t flag;
    };

    int fail(const std::string& message);
    int read_file(const std::string& name, const StoreZipMeta& meta, char* data);

    FILE* fp;
    uint64_t data_limit;
    std::map<std::string, StoreZipMeta> filemetas;
};

class StoreZipWriter
{
public:
    StoreZipWriter();
    ~StoreZipWriter();

    int open(const std::string& path);

    int write_file(const std::string& name, const char* data, uint64_t size);

    int close();

private:
    FILE* fp;

    struct StoreZipMeta
    {
        std::string name;
        uint64_t lfh_offset;
        uint32_t crc32;
        uint64_t size;
    };

    std::vector<StoreZipMeta> filemetas;
};

} // namespace pnnx

#endif // PNNX_STOREZIP_H
