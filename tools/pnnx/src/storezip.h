// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_STOREZIP_H
#define PNNX_STOREZIP_H

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

    int close();

private:
    FILE* fp;

    struct StoreZipMeta
    {
        uint64_t offset;
        uint64_t size;
    };

    std::map<std::string, StoreZipMeta> filemetas;

    // Locate the central directory via the End Of Central Directory record.
    // Returns 0 on success and fills cd_offset / cd_size / cd_records. This is
    // the authoritative index of the zip and is immune to data descriptors.
    int find_central_directory(uint64_t& cd_offset, uint64_t& cd_size, uint64_t& cd_records);

    // Returns true if a central directory file header signature starts at off.
    bool cd_offset_valid(uint64_t off);
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
