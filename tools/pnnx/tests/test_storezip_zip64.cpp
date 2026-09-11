// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "storezip.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <io.h>
#else
#include <sys/types.h>
#endif

static int seek64(FILE* fp, int64_t offset, int origin)
{
#ifdef _WIN32
    return _fseeki64(fp, offset, origin);
#else
    return fseeko(fp, (off_t)offset, origin);
#endif
}

static bool write_bytes(FILE* fp, const void* data, size_t size)
{
    return fwrite(data, 1, size, fp) == size;
}

static bool write_le16(FILE* fp, uint16_t value)
{
    unsigned char bytes[2] = {(unsigned char)value, (unsigned char)(value >> 8)};
    return write_bytes(fp, bytes, sizeof(bytes));
}

static bool write_le32(FILE* fp, uint32_t value)
{
    unsigned char bytes[4] = {(unsigned char)value, (unsigned char)(value >> 8), (unsigned char)(value >> 16), (unsigned char)(value >> 24)};
    return write_bytes(fp, bytes, sizeof(bytes));
}

static bool write_le64(FILE* fp, uint64_t value)
{
    unsigned char bytes[8];
    for (int i = 0; i < 8; i++)
        bytes[i] = (unsigned char)(value >> (8 * i));
    return write_bytes(fp, bytes, sizeof(bytes));
}

static bool write_local_header(FILE* fp)
{
    return write_le32(fp, 0x04034b50) && write_le16(fp, 45) && write_le16(fp, 0) && write_le16(fp, 0)
           && write_le16(fp, 0) && write_le16(fp, 0) && write_le32(fp, 0) && write_le32(fp, 0)
           && write_le32(fp, 0) && write_le16(fp, 1) && write_le16(fp, 0) && write_bytes(fp, "x", 1);
}

static bool write_central_header(FILE* fp, uint64_t central_offset, uint64_t central_size)
{
    if (!write_le32(fp, 0x02014b50) || !write_le16(fp, 45) || !write_le16(fp, 45) || !write_le16(fp, 0)
            || !write_le16(fp, 0) || !write_le16(fp, 0) || !write_le16(fp, 0) || !write_le32(fp, 0)
            || !write_le32(fp, 0) || !write_le32(fp, 0) || !write_le16(fp, 1) || !write_le16(fp, 28)
            || !write_le16(fp, 0) || !write_le16(fp, 0) || !write_le16(fp, 0) || !write_le32(fp, 0)
            || !write_le32(fp, 0xffffffff) || !write_bytes(fp, "x", 1) || !write_le16(fp, 1) || !write_le16(fp, 24)
            || !write_le64(fp, 0) || !write_le64(fp, 0) || !write_le64(fp, 0))
        return false;

    return central_offset > 0 && central_size > 0;
}

static bool write_end_records(FILE* fp, uint64_t central_offset, uint64_t central_size, uint64_t eocdr64_offset)
{
    return write_le32(fp, 0x06064b50) && write_le64(fp, 44) && write_le16(fp, 45) && write_le16(fp, 45)
           && write_le32(fp, 0) && write_le32(fp, 0) && write_le64(fp, 1) && write_le64(fp, 1)
           && write_le64(fp, central_size) && write_le64(fp, central_offset) && write_le32(fp, 0x07064b50)
           && write_le32(fp, 0) && write_le64(fp, eocdr64_offset) && write_le32(fp, 1) && write_le32(fp, 0x06054b50)
           && write_le16(fp, 0) && write_le16(fp, 0) && write_le16(fp, 0xffff) && write_le16(fp, 0xffff)
           && write_le32(fp, 0xffffffff) && write_le32(fp, 0xffffffff) && write_le16(fp, 0);
}

int main()
{
    const char* path = "test_storezip_zip64_sparse.pt2";
    const uint64_t central_offset = (uint64_t)2 * 1024 * 1024 * 1024 + 4096;
    const uint64_t central_size = 4 + 42 + 1 + 4 + 24;
    const uint64_t eocdr64_offset = central_offset + central_size;

    FILE* fp = fopen(path, "wb");
    if (!fp || !write_local_header(fp) || seek64(fp, (int64_t)central_offset, SEEK_SET) != 0
            || !write_central_header(fp, central_offset, central_size)
            || !write_end_records(fp, central_offset, central_size, eocdr64_offset))
    {
        if (fp)
            fclose(fp);
        remove(path);
        return 1;
    }
    fclose(fp);

    pnnx::StoreZipReader reader;
    const int opened = reader.open(path);
    const std::vector<std::string> names = reader.get_names();
    const bool valid = opened == 0 && names.size() == 1 && names[0] == "x" && reader.get_file_size("x") == 0;
    reader.close();
    remove(path);

    if (!valid)
    {
        fprintf(stderr, "Zip64 sparse archive validation failed\n");
        return 1;
    }

    printf("Zip64 sparse archive validation passed at offset %llu\n", (unsigned long long)central_offset);
    return 0;
}
