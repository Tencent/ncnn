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

    // Metadata queries do not read or decompress records; absent names return -1.
    int get_file_compression(const std::string& name) const;

    int get_file_flags(const std::string& name) const;

    uint64_t get_file_size(const std::string& name) const;

    // Physical file size measured when opening the archive; zero when closed.
    uint64_t get_archive_size() const;

    int read_file(const std::string& name, char* data);

    int close();

private:
    int open_archive();

    FILE* fp;
    uint64_t archive_size;

    struct StoreZipMeta
    {
        uint64_t offset;
        uint64_t size;
        uint32_t crc32;
        uint16_t flags;
        uint16_t compression;
    };

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
    int write_central_directory();

    FILE* fp;
    bool failed;

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
