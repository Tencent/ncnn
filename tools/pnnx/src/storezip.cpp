// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "storezip.h"

#include <array>
#include <stdio.h>
#include <stdint.h>
#if !defined(_WIN32)
#include <sys/types.h>
#endif
#include <limits>
#include <map>
#include <new>
#include <set>
#include <string>
#include <stdexcept>
#include <vector>

namespace pnnx {

// https://stackoverflow.com/questions/1537964/visual-c-equivalent-of-gccs-attribute-packed
#ifdef _MSC_VER
#define PACK(__Declaration__) __pragma(pack(push, 1)) __Declaration__ __pragma(pack(pop))
#else
#define PACK(__Declaration__) __Declaration__ __attribute__((__packed__))
#endif

PACK(struct local_file_header {
    uint16_t version;
    uint16_t flag;
    uint16_t compression;
    uint16_t last_modify_time;
    uint16_t last_modify_date;
    uint32_t crc32;
    uint32_t compressed_size;
    uint32_t uncompressed_size;
    uint16_t file_name_length;
    uint16_t extra_field_length;
});

PACK(struct zip64_extended_extra_field {
    uint64_t uncompressed_size;
    uint64_t compressed_size;
    uint64_t lfh_offset;
    uint32_t disk_number;
});

PACK(struct central_directory_file_header {
    uint16_t version_made;
    uint16_t version;
    uint16_t flag;
    uint16_t compression;
    uint16_t last_modify_time;
    uint16_t last_modify_date;
    uint32_t crc32;
    uint32_t compressed_size;
    uint32_t uncompressed_size;
    uint16_t file_name_length;
    uint16_t extra_field_length;
    uint16_t file_comment_length;
    uint16_t start_disk;
    uint16_t internal_file_attrs;
    uint32_t external_file_attrs;
    uint32_t lfh_offset;
});

PACK(struct zip64_end_of_central_directory_record {
    uint64_t size_of_eocd64_m12;
    uint16_t version_made_by;
    uint16_t version_min_required;
    uint32_t disk_number;
    uint32_t start_disk;
    uint64_t cd_records;
    uint64_t total_cd_records;
    uint64_t cd_size;
    uint64_t cd_offset;
});

PACK(struct zip64_end_of_central_directory_locator {
    uint32_t eocdr64_disk_number;
    uint64_t eocdr64_offset;
    uint32_t disk_count;
});

PACK(struct end_of_central_directory_record {
    uint16_t disk_number;
    uint16_t start_disk;
    uint16_t cd_records;
    uint16_t total_cd_records;
    uint32_t cd_size;
    uint32_t cd_offset;
    uint16_t comment_length;
});

static const std::array<uint32_t, 256>& CRC32_TABLE()
{
    static const std::array<uint32_t, 256> table = []() {
        std::array<uint32_t, 256> values = {};
        for (int i = 0; i < 256; i++)
        {
            uint32_t c = i;
            for (int j = 0; j < 8; j++)
            {
                if (c & 1)
                    c = (c >> 1) ^ 0xedb88320;
                else
                    c >>= 1;
            }
            values[i] = c;
        }
        return values;
    }
    ();

    return table;
}

static uint32_t CRC32(uint32_t x, unsigned char ch)
{
    return (x >> 8) ^ CRC32_TABLE()[(x ^ ch) & 0xff];
}

static uint32_t CRC32_buffer(const unsigned char* data, uint64_t len)
{
    uint32_t x = 0xffffffff;

    for (uint64_t i = 0; i < len; i++)
        x = CRC32(x, data[i]);

    return x ^ 0xffffffff;
}

static uint16_t read_le16(const unsigned char* p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t read_le32(const unsigned char* p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t read_le64(const unsigned char* p)
{
    return (uint64_t)read_le32(p) | ((uint64_t)read_le32(p + 4) << 32);
}

static int storezip_fseek(FILE* fp, int64_t offset, int origin)
{
#if defined(_WIN32)
    return _fseeki64(fp, offset, origin);
#else
    const off_t offset2 = (off_t)offset;
    if ((int64_t)offset2 != offset)
        return -1;

    return fseeko(fp, offset2, origin);
#endif
}

static int64_t storezip_ftell(FILE* fp)
{
#if defined(_WIN32)
    return _ftelli64(fp);
#else
    return (int64_t)ftello(fp);
#endif
}

static int storezip_fseek_absolute(FILE* fp, uint64_t offset)
{
    if (offset > (uint64_t)std::numeric_limits<int64_t>::max())
        return -1;

    return storezip_fseek(fp, (int64_t)offset, SEEK_SET);
}

static bool storezip_fread_exact(FILE* fp, void* data, size_t size)
{
    if (size == 0)
        return true;

    return fread(data, 1, size, fp) == size;
}

struct StoreZipCentralEntry
{
    std::string name;
    uint64_t compressed_size;
    uint64_t uncompressed_size;
    uint64_t local_header_offset;
    uint32_t crc32;
    uint16_t flag;
    uint16_t compression;
};

StoreZipReader::StoreZipReader()
{
    fp = 0;
}

StoreZipReader::~StoreZipReader()
{
    close();
}

int StoreZipReader::open(const std::string& path)
try
{
    close();

    fp = fopen(path.c_str(), "rb");
    if (!fp)
    {
        fprintf(stderr, "open failed\n");
        return -1;
    }

    const int fail_open = -1;

    if (storezip_fseek(fp, 0, SEEK_END) != 0)
    {
        fprintf(stderr, "seek zip file failed\n");
        close();
        return fail_open;
    }

    const int64_t file_size_signed = storezip_ftell(fp);
    if (file_size_signed < 0)
    {
        fprintf(stderr, "tell zip file failed\n");
        close();
        return fail_open;
    }

    const uint64_t file_size = (uint64_t)file_size_signed;
    const uint64_t max_eocd_search_size = 22 + 65535;
    const size_t eocd_search_size = (size_t)(file_size < max_eocd_search_size ? file_size : max_eocd_search_size);
    if (eocd_search_size < 22 || storezip_fseek_absolute(fp, file_size - eocd_search_size) != 0)
    {
        fprintf(stderr, "zip end of central directory not found\n");
        close();
        return fail_open;
    }

    std::vector<unsigned char> eocd_search(eocd_search_size);
    if (!storezip_fread_exact(fp, eocd_search.data(), eocd_search.size()))
    {
        fprintf(stderr, "read zip end of central directory failed\n");
        close();
        return fail_open;
    }

    size_t eocd_search_offset = eocd_search_size;
    for (size_t i = eocd_search_size - 22 + 1; i != 0;)
    {
        i--;

        if (read_le32(eocd_search.data() + i) != 0x06054b50)
            continue;

        const uint16_t comment_length = read_le16(eocd_search.data() + i + 20);
        if (i + 22 + comment_length == eocd_search_size)
        {
            eocd_search_offset = i;
            break;
        }
    }

    if (eocd_search_offset == eocd_search_size)
    {
        fprintf(stderr, "zip end of central directory not found\n");
        close();
        return fail_open;
    }

    const unsigned char* eocd = eocd_search.data() + eocd_search_offset;
    const uint64_t eocd_offset = file_size - eocd_search_size + eocd_search_offset;
    const uint16_t eocd_disk_number = read_le16(eocd + 4);
    const uint16_t eocd_start_disk = read_le16(eocd + 6);
    const uint16_t eocd_disk_records = read_le16(eocd + 8);
    const uint16_t eocd_total_records = read_le16(eocd + 10);
    const uint32_t eocd_cd_size = read_le32(eocd + 12);
    const uint32_t eocd_cd_offset = read_le32(eocd + 16);

    const bool zip64 = eocd_disk_number == 0xffff || eocd_start_disk == 0xffff || eocd_disk_records == 0xffff || eocd_total_records == 0xffff || eocd_cd_size == 0xffffffff || eocd_cd_offset == 0xffffffff;

    uint64_t total_records = eocd_total_records;
    uint64_t cd_size = eocd_cd_size;
    uint64_t cd_offset = eocd_cd_offset;
    uint64_t cd_end_limit = eocd_offset;

    if (zip64)
    {
        if (eocd_offset < 20 || storezip_fseek_absolute(fp, eocd_offset - 20) != 0)
        {
            fprintf(stderr, "zip64 end of central directory locator not found\n");
            close();
            return fail_open;
        }

        unsigned char locator[20];
        if (!storezip_fread_exact(fp, locator, sizeof(locator)) || read_le32(locator) != 0x07064b50)
        {
            fprintf(stderr, "zip64 end of central directory locator not found\n");
            close();
            return fail_open;
        }

        const uint64_t locator_offset = eocd_offset - 20;
        const uint32_t zip64_disk_number = read_le32(locator + 4);
        const uint64_t zip64_eocd_offset = read_le64(locator + 8);
        const uint32_t zip64_disk_count = read_le32(locator + 16);
        if (zip64_disk_number != 0 || zip64_disk_count != 1 || zip64_eocd_offset > locator_offset || storezip_fseek_absolute(fp, zip64_eocd_offset) != 0)
        {
            fprintf(stderr, "multi-disk or invalid zip64 archive is not supported\n");
            close();
            return fail_open;
        }

        unsigned char eocd64[56];
        if (!storezip_fread_exact(fp, eocd64, sizeof(eocd64)) || read_le32(eocd64) != 0x06064b50)
        {
            fprintf(stderr, "invalid zip64 end of central directory\n");
            close();
            return fail_open;
        }

        const uint64_t zip64_eocd_gap = locator_offset - zip64_eocd_offset;
        const uint64_t eocd64_size = read_le64(eocd64 + 4);
        if (zip64_eocd_gap < sizeof(eocd64) || eocd64_size < 44 || eocd64_size > zip64_eocd_gap - 12)
        {
            fprintf(stderr, "invalid zip64 end of central directory metadata\n");
            close();
            return fail_open;
        }

        const uint32_t disk_number = read_le32(eocd64 + 16);
        const uint32_t start_disk = read_le32(eocd64 + 20);
        const uint64_t disk_records = read_le64(eocd64 + 24);
        total_records = read_le64(eocd64 + 32);
        cd_size = read_le64(eocd64 + 40);
        cd_offset = read_le64(eocd64 + 48);
        cd_end_limit = zip64_eocd_offset;

        if (disk_number != 0 || start_disk != 0 || disk_records != total_records)
        {
            fprintf(stderr, "multi-disk zip archive is not supported\n");
            close();
            return fail_open;
        }
    }
    else if (eocd_disk_number != 0 || eocd_start_disk != 0 || eocd_disk_records != eocd_total_records)
    {
        fprintf(stderr, "multi-disk zip archive is not supported\n");
        close();
        return fail_open;
    }

    if (cd_offset > cd_end_limit || cd_size > cd_end_limit - cd_offset || total_records > cd_size / 46 || total_records > (uint64_t)std::numeric_limits<size_t>::max())
    {
        fprintf(stderr, "invalid zip central directory range\n");
        close();
        return fail_open;
    }

    const uint64_t cd_end = cd_offset + cd_size;
    uint64_t cursor = cd_offset;
    std::vector<StoreZipCentralEntry> entries;
    entries.reserve((size_t)total_records);
    std::set<std::string> entry_names;

    for (uint64_t i = 0; i < total_records; i++)
    {
        if (cursor > cd_end || 46 > cd_end - cursor || storezip_fseek_absolute(fp, cursor) != 0)
        {
            fprintf(stderr, "truncated zip central directory\n");
            close();
            return fail_open;
        }

        unsigned char cdfh[46];
        if (!storezip_fread_exact(fp, cdfh, sizeof(cdfh)) || read_le32(cdfh) != 0x02014b50)
        {
            fprintf(stderr, "invalid zip central directory file header\n");
            close();
            return fail_open;
        }

        const uint16_t flag = read_le16(cdfh + 8);
        const uint16_t compression = read_le16(cdfh + 10);
        const uint32_t crc32 = read_le32(cdfh + 16);
        uint64_t compressed_size = read_le32(cdfh + 20);
        uint64_t uncompressed_size = read_le32(cdfh + 24);
        const uint16_t file_name_length = read_le16(cdfh + 28);
        const uint16_t extra_field_length = read_le16(cdfh + 30);
        const uint16_t file_comment_length = read_le16(cdfh + 32);
        uint64_t start_disk = read_le16(cdfh + 34);
        uint64_t local_header_offset = read_le32(cdfh + 42);

        const uint64_t variable_size = (uint64_t)file_name_length + extra_field_length + file_comment_length;
        if (variable_size > cd_end - cursor - 46)
        {
            fprintf(stderr, "truncated zip central directory entry\n");
            close();
            return fail_open;
        }

        std::string name(file_name_length, '\0');
        std::vector<unsigned char> extra(extra_field_length);
        if (!storezip_fread_exact(fp, name.empty() ? 0 : &name[0], name.size()) || !storezip_fread_exact(fp, extra.data(), extra.size()))
        {
            fprintf(stderr, "read zip central directory entry failed\n");
            close();
            return fail_open;
        }

        const bool need_uncompressed_size = uncompressed_size == 0xffffffff;
        const bool need_compressed_size = compressed_size == 0xffffffff;
        const bool need_local_header_offset = local_header_offset == 0xffffffff;
        const bool need_start_disk = start_disk == 0xffff;
        bool found_zip64_extra = false;

        size_t extra_offset = 0;
        while (extra_offset < extra.size())
        {
            if (extra.size() - extra_offset < 4)
            {
                fprintf(stderr, "invalid zip extra field\n");
                close();
                return fail_open;
            }

            const uint16_t extra_id = read_le16(extra.data() + extra_offset);
            const uint16_t extra_size = read_le16(extra.data() + extra_offset + 2);
            extra_offset += 4;
            if (extra_size > extra.size() - extra_offset)
            {
                fprintf(stderr, "invalid zip extra field size\n");
                close();
                return fail_open;
            }

            if (extra_id == 0x0001)
            {
                found_zip64_extra = true;
                size_t zip64_offset = extra_offset;
                const size_t zip64_end = extra_offset + extra_size;

                if (need_uncompressed_size)
                {
                    if (zip64_end - zip64_offset < 8)
                    {
                        fprintf(stderr, "invalid zip64 uncompressed size\n");
                        close();
                        return fail_open;
                    }
                    uncompressed_size = read_le64(extra.data() + zip64_offset);
                    zip64_offset += 8;
                }
                if (need_compressed_size)
                {
                    if (zip64_end - zip64_offset < 8)
                    {
                        fprintf(stderr, "invalid zip64 compressed size\n");
                        close();
                        return fail_open;
                    }
                    compressed_size = read_le64(extra.data() + zip64_offset);
                    zip64_offset += 8;
                }
                if (need_local_header_offset)
                {
                    if (zip64_end - zip64_offset < 8)
                    {
                        fprintf(stderr, "invalid zip64 local header offset\n");
                        close();
                        return fail_open;
                    }
                    local_header_offset = read_le64(extra.data() + zip64_offset);
                    zip64_offset += 8;
                }
                if (need_start_disk)
                {
                    if (zip64_end - zip64_offset < 4)
                    {
                        fprintf(stderr, "invalid zip64 start disk\n");
                        close();
                        return fail_open;
                    }
                    start_disk = read_le32(extra.data() + zip64_offset);
                }
            }

            extra_offset += extra_size;
        }

        if ((need_uncompressed_size || need_compressed_size || need_local_header_offset || need_start_disk) && !found_zip64_extra)
        {
            fprintf(stderr, "zip64 extra field not found\n");
            close();
            return fail_open;
        }

        if ((flag & 0x0041) != 0)
        {
            fprintf(stderr, "encrypted zip entry is not supported\n");
            close();
            return fail_open;
        }
        if (compression == 0 && compressed_size != uncompressed_size)
        {
            fprintf(stderr, "stored zip entry has different compressed and uncompressed sizes\n");
            close();
            return fail_open;
        }
        if (start_disk != 0)
        {
            fprintf(stderr, "multi-disk zip entry is not supported\n");
            close();
            return fail_open;
        }
        if (!entry_names.insert(name).second)
        {
            fprintf(stderr, "duplicate zip entry name %s\n", name.c_str());
            close();
            return fail_open;
        }

        StoreZipCentralEntry entry;
        entry.name = name;
        entry.compressed_size = compressed_size;
        entry.uncompressed_size = uncompressed_size;
        entry.local_header_offset = local_header_offset;
        entry.crc32 = crc32;
        entry.flag = flag;
        entry.compression = compression;
        entries.push_back(entry);

        cursor += 46 + variable_size;
    }

    if (cursor != cd_end)
    {
        fprintf(stderr, "unsupported data after zip central directory entries\n");
        close();
        return fail_open;
    }

    std::map<std::string, StoreZipMeta> filemetas2;
    for (size_t i = 0; i < entries.size(); i++)
    {
        const StoreZipCentralEntry& entry = entries[i];
        if (entry.local_header_offset > cd_offset || 30 > cd_offset - entry.local_header_offset || storezip_fseek_absolute(fp, entry.local_header_offset) != 0)
        {
            fprintf(stderr, "invalid zip local file header offset\n");
            close();
            return fail_open;
        }

        unsigned char lfh[30];
        if (!storezip_fread_exact(fp, lfh, sizeof(lfh)) || read_le32(lfh) != 0x04034b50)
        {
            fprintf(stderr, "invalid zip local file header\n");
            close();
            return fail_open;
        }

        const uint16_t local_flag = read_le16(lfh + 6);
        const uint16_t local_compression = read_le16(lfh + 8);
        const uint32_t local_crc32 = read_le32(lfh + 14);
        const uint32_t local_compressed_size = read_le32(lfh + 18);
        const uint32_t local_uncompressed_size = read_le32(lfh + 22);
        const uint16_t local_file_name_length = read_le16(lfh + 26);
        const uint16_t local_extra_field_length = read_le16(lfh + 28);
        const uint64_t local_variable_size = (uint64_t)local_file_name_length + local_extra_field_length;

        if ((local_flag & 0x0041) != 0)
        {
            fprintf(stderr, "encrypted zip local file entry is not supported\n");
            close();
            return fail_open;
        }
        if (local_flag != entry.flag)
        {
            fprintf(stderr, "zip local file flag mismatch %s\n", entry.name.c_str());
            close();
            return fail_open;
        }
        if (local_compression != entry.compression)
        {
            fprintf(stderr, "zip local file compression mismatch %s\n", entry.name.c_str());
            close();
            return fail_open;
        }
        if ((local_flag & 0x0008) == 0 && local_crc32 != entry.crc32)
        {
            fprintf(stderr, "zip local file crc mismatch %s\n", entry.name.c_str());
            close();
            return fail_open;
        }
        const bool local_compressed_size_mismatch = local_compressed_size != 0xffffffff && local_compressed_size != entry.compressed_size;
        const bool local_uncompressed_size_mismatch = local_uncompressed_size != 0xffffffff && local_uncompressed_size != entry.uncompressed_size;
        if ((local_flag & 0x0008) == 0 && (local_compressed_size_mismatch || local_uncompressed_size_mismatch))
        {
            fprintf(stderr, "zip local file size mismatch %s\n", entry.name.c_str());
            close();
            return fail_open;
        }
        if (local_variable_size > cd_offset - entry.local_header_offset - 30)
        {
            fprintf(stderr, "truncated zip local file header\n");
            close();
            return fail_open;
        }

        std::string local_name(local_file_name_length, '\0');
        if (!storezip_fread_exact(fp, local_name.empty() ? 0 : &local_name[0], local_name.size()))
        {
            fprintf(stderr, "read zip local file name failed\n");
            close();
            return fail_open;
        }
        if (local_name != entry.name)
        {
            fprintf(stderr, "zip local file name mismatch %s\n", entry.name.c_str());
            close();
            return fail_open;
        }

        const uint64_t data_offset = entry.local_header_offset + 30 + local_variable_size;
        if (data_offset > cd_offset || entry.compressed_size > cd_offset - data_offset)
        {
            fprintf(stderr, "zip entry data is outside archive data range\n");
            close();
            return fail_open;
        }

        StoreZipMeta fm;
        fm.offset = data_offset;
        fm.compressed_size = entry.compressed_size;
        fm.uncompressed_size = entry.uncompressed_size;
        fm.crc32 = entry.crc32;
        fm.flag = entry.flag;
        fm.compression = entry.compression;
        filemetas2[entry.name] = fm;
    }

    filemetas.swap(filemetas2);

    return 0;
}
catch (const std::length_error&)
{
    close();
    fprintf(stderr, "archive allocation failed\n");
    return -1;
}
catch (const std::bad_alloc&)
{
    close();
    fprintf(stderr, "archive allocation failed\n");
    return -1;
}

int StoreZipReader::get_names(std::vector<std::string>& names) const
try
{
    std::vector<std::string> names2;
    names2.reserve(filemetas.size());
    for (std::map<std::string, StoreZipMeta>::const_iterator it = filemetas.begin(); it != filemetas.end(); ++it)
    {
        names2.push_back(it->first);
    }

    names.swap(names2);
    return 0;
}
catch (const std::length_error&)
{
    fprintf(stderr, "archive allocation failed\n");
    return -1;
}
catch (const std::bad_alloc&)
{
    fprintf(stderr, "archive allocation failed\n");
    return -1;
}

bool StoreZipReader::has_file(const std::string& name) const
{
    return filemetas.find(name) != filemetas.end();
}

bool StoreZipReader::is_file_stored(const std::string& name) const
{
    std::map<std::string, StoreZipMeta>::const_iterator it = filemetas.find(name);
    if (it == filemetas.end())
    {
        fprintf(stderr, "no such file %s\n", name.c_str());
        return false;
    }

    return it->second.compression == 0;
}

uint64_t StoreZipReader::get_file_size(const std::string& name) const
{
    std::map<std::string, StoreZipMeta>::const_iterator it = filemetas.find(name);
    if (it == filemetas.end())
    {
        fprintf(stderr, "no such file %s\n", name.c_str());
        return 0;
    }

    return it->second.uncompressed_size;
}

int StoreZipReader::read_file(const std::string& name, char* data)
{
    if (!fp)
    {
        fprintf(stderr, "zip file is not open\n");
        return -1;
    }

    std::map<std::string, StoreZipMeta>::const_iterator it = filemetas.find(name);
    if (it == filemetas.end())
    {
        fprintf(stderr, "no such file %s\n", name.c_str());
        return -1;
    }

    const StoreZipMeta& fm = it->second;
    if (fm.compression != 0)
    {
        fprintf(stderr, "compressed zip entry is not supported %s method=%u\n", name.c_str(), (unsigned int)fm.compression);
        return -1;
    }

    uint64_t size = fm.uncompressed_size;

    if (storezip_fseek_absolute(fp, fm.offset) != 0)
    {
        fprintf(stderr, "seek zip entry failed %s\n", name.c_str());
        return -1;
    }

    uint32_t crc32 = 0xffffffff;
    while (size != 0)
    {
        const size_t chunk_size = size > 0x40000000 ? 0x40000000 : (size_t)size;
        if (!storezip_fread_exact(fp, data, chunk_size))
        {
            fprintf(stderr, "read zip entry failed %s\n", name.c_str());
            return -1;
        }

        for (size_t i = 0; i < chunk_size; i++)
            crc32 = CRC32(crc32, (unsigned char)data[i]);

        data += chunk_size;
        size -= chunk_size;
    }

    crc32 ^= 0xffffffff;
    if (crc32 != fm.crc32)
    {
        fprintf(stderr, "zip entry crc mismatch %s\n", name.c_str());
        return -1;
    }

    return 0;
}

int StoreZipReader::close()
{
    if (fp)
        fclose(fp);
    fp = 0;
    filemetas.clear();

    return 0;
}

StoreZipWriter::StoreZipWriter()
{
    fp = 0;
}

StoreZipWriter::~StoreZipWriter()
{
    close();
}

int StoreZipWriter::open(const std::string& path)
{
    close();

    fp = fopen(path.c_str(), "wb");
    if (!fp)
    {
        fprintf(stderr, "open failed\n");
        return -1;
    }

    return 0;
}

int StoreZipWriter::write_file(const std::string& name, const char* data, uint64_t size)
{
    const int64_t offset_signed = storezip_ftell(fp);
    if (offset_signed < 0)
    {
        fprintf(stderr, "tell zip file failed\n");
        return -1;
    }
    const uint64_t offset = (uint64_t)offset_signed;

    uint32_t signature = 0x04034b50;
    fwrite((char*)&signature, sizeof(signature), 1, fp);

    uint32_t crc32 = CRC32_buffer((const unsigned char*)data, size);

    local_file_header lfh;
    lfh.version = 0;
    lfh.flag = 0;
    lfh.compression = 0;
    lfh.last_modify_time = 0;
    lfh.last_modify_date = 0;
    lfh.crc32 = crc32;
    lfh.compressed_size = 0xffffffff;
    lfh.uncompressed_size = 0xffffffff;
    lfh.file_name_length = name.size();

    // zip64 extra field
    zip64_extended_extra_field zip64_eef;
    zip64_eef.uncompressed_size = size;
    zip64_eef.compressed_size = size;
    zip64_eef.lfh_offset = 0;
    zip64_eef.disk_number = 0;

    uint16_t extra_id = 0x0001;
    uint16_t extra_size = sizeof(zip64_eef);

    lfh.extra_field_length = sizeof(extra_id) + sizeof(extra_size) + sizeof(zip64_eef);

    fwrite((char*)&lfh, sizeof(lfh), 1, fp);

    fwrite((char*)name.c_str(), name.size(), 1, fp);

    fwrite((char*)&extra_id, sizeof(extra_id), 1, fp);
    fwrite((char*)&extra_size, sizeof(extra_size), 1, fp);
    fwrite((char*)&zip64_eef, sizeof(zip64_eef), 1, fp);

    fwrite(data, size, 1, fp);

    StoreZipMeta szm;
    szm.name = name;
    szm.lfh_offset = offset;
    szm.crc32 = crc32;
    szm.size = size;

    filemetas.push_back(szm);

    return 0;
}

int StoreZipWriter::close()
{
    if (!fp)
        return 0;

    const int64_t offset_signed = storezip_ftell(fp);
    if (offset_signed < 0)
    {
        fprintf(stderr, "tell zip file failed\n");
        fclose(fp);
        fp = 0;
        filemetas.clear();
        return -1;
    }
    const uint64_t offset = (uint64_t)offset_signed;

    for (const StoreZipMeta& szm : filemetas)
    {
        uint32_t signature = 0x02014b50;
        fwrite((char*)&signature, sizeof(signature), 1, fp);

        central_directory_file_header cdfh;
        cdfh.version_made = 0;
        cdfh.version = 0;
        cdfh.flag = 0;
        cdfh.compression = 0;
        cdfh.last_modify_time = 0;
        cdfh.last_modify_date = 0;
        cdfh.crc32 = szm.crc32;
        cdfh.compressed_size = 0xffffffff;
        cdfh.uncompressed_size = 0xffffffff;
        cdfh.file_name_length = szm.name.size();
        cdfh.file_comment_length = 0;
        cdfh.start_disk = 0xffff;
        cdfh.internal_file_attrs = 0;
        cdfh.external_file_attrs = 0;
        cdfh.lfh_offset = 0xffffffff;

        // zip64 extra field
        zip64_extended_extra_field zip64_eef;
        zip64_eef.uncompressed_size = szm.size;
        zip64_eef.compressed_size = szm.size;
        zip64_eef.lfh_offset = szm.lfh_offset;
        zip64_eef.disk_number = 0;

        uint16_t extra_id = 0x0001;
        uint16_t extra_size = sizeof(zip64_eef);

        cdfh.extra_field_length = sizeof(extra_id) + sizeof(extra_size) + sizeof(zip64_eef);

        fwrite((char*)&cdfh, sizeof(cdfh), 1, fp);

        fwrite((char*)szm.name.c_str(), szm.name.size(), 1, fp);

        fwrite((char*)&extra_id, sizeof(extra_id), 1, fp);
        fwrite((char*)&extra_size, sizeof(extra_size), 1, fp);
        fwrite((char*)&zip64_eef, sizeof(zip64_eef), 1, fp);
    }

    const int64_t offset2_signed = storezip_ftell(fp);
    if (offset2_signed < 0)
    {
        fprintf(stderr, "tell zip file failed\n");
        fclose(fp);
        fp = 0;
        filemetas.clear();
        return -1;
    }
    const uint64_t offset2 = (uint64_t)offset2_signed;

    {
        uint32_t signature = 0x06064b50;
        fwrite((char*)&signature, sizeof(signature), 1, fp);

        zip64_end_of_central_directory_record eocdr64;
        eocdr64.size_of_eocd64_m12 = sizeof(eocdr64) - 8;
        eocdr64.version_made_by = 0;
        eocdr64.version_min_required = 0;
        eocdr64.disk_number = 0;
        eocdr64.start_disk = 0;
        eocdr64.cd_records = filemetas.size();
        eocdr64.total_cd_records = filemetas.size();
        eocdr64.cd_size = offset2 - offset;
        eocdr64.cd_offset = offset;

        fwrite((char*)&eocdr64, sizeof(eocdr64), 1, fp);
    }

    {
        uint32_t signature = 0x07064b50;
        fwrite((char*)&signature, sizeof(signature), 1, fp);

        zip64_end_of_central_directory_locator eocdl64;
        eocdl64.eocdr64_disk_number = 0;
        eocdl64.eocdr64_offset = offset2;
        eocdl64.disk_count = 1;

        fwrite((char*)&eocdl64, sizeof(eocdl64), 1, fp);
    }

    {
        uint32_t signature = 0x06054b50;
        fwrite((char*)&signature, sizeof(signature), 1, fp);

        end_of_central_directory_record eocdr;
        eocdr.disk_number = 0xffff;
        eocdr.start_disk = 0xffff;
        eocdr.cd_records = 0xffff;
        eocdr.total_cd_records = 0xffff;
        eocdr.cd_size = 0xffffffff;
        eocdr.cd_offset = 0xffffffff;
        eocdr.comment_length = 0;

        fwrite((char*)&eocdr, sizeof(eocdr), 1, fp);
    }

    fclose(fp);
    fp = 0;

    return 0;
}

} // namespace pnnx

#if 0
int main()
{
    using namespace pnnx;

    {
        uint64_t len = 1*1024*1024*1024;
        // uint64_t len = 1*1024*1024;
        char* data1g = new char[len];

        StoreZipWriter szw;

        szw.open("szw.zip");

        szw.write_file("a.py", data1g, len);
        szw.write_file("b.param", data1g, 44);
        szw.write_file("c.bin", data1g, len);
        szw.write_file("d.txt", data1g, len);
        szw.write_file("e.jpg", data1g, len);
        szw.write_file("f.png", data1g, len);

        szw.close();

        delete[] data1g;
    }

    {
        StoreZipReader sz;

        sz.open("szw.zip");

        std::vector<std::string> names;
        if (sz.get_names(names) != 0)
            return -1;

        for (size_t i = 0; i < names.size(); i++)
        {
            uint64_t size = sz.get_file_size(names[i]);

            fprintf(stderr, "%s  %lu\n", names[i].c_str(), size);
        }

        sz.close();
    }

    return 0;
}
#endif
