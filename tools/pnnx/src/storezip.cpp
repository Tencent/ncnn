// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Keep fseeko/ftello 64-bit on 32-bit POSIX builds as well.
#ifndef _WIN32
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif
#endif

#include "storezip.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <limits>
#include <map>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
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

struct CRC32Table
{
    CRC32Table()
    {
        for (int i = 0; i < 256; i++)
        {
            uint32_t c = i;
            for (int j = 0; j < 8; j++)
                c = (c >> 1) ^ ((c & 1) ? 0xedb88320u : 0);
            values[i] = c;
        }
    }

    uint32_t values[256];
};

static uint32_t CRC32_buffer(const unsigned char* data, size_t len)
{
    // C++11 initialization is thread-safe and does not depend on a writer
    // having been constructed before the first reader.
    static const CRC32Table table;
    uint32_t x = 0xffffffff;

    for (size_t i = 0; i < len; i++)
        x = (x >> 8) ^ table.values[(x ^ data[i]) & 0xff];

    return x ^ 0xffffffff;
}

StoreZipReader::StoreZipReader()
{
    fp = 0;
    archive_size = 0;
}

StoreZipReader::~StoreZipReader()
{
    close();
}

static int file_seek(FILE* fp, int64_t offset, int origin)
{
#if _WIN32
    return _fseeki64(fp, offset, origin);
#else
    if ((int64_t)(off_t)offset != offset)
        return -1;
    return fseeko(fp, (off_t)offset, origin);
#endif
}

static int64_t file_tell(FILE* fp)
{
#if _WIN32
    return _ftelli64(fp);
#else
    return ftello(fp);
#endif
}

static bool read_bytes(FILE* fp, void* data, size_t size)
{
    return size == 0 || fread(data, 1, size, fp) == size;
}

static bool write_bytes(FILE* fp, const void* data, size_t size)
{
    return size == 0 || fwrite(data, 1, size, fp) == size;
}

static bool zip_range_within(uint64_t offset, uint64_t size, uint64_t limit)
{
    // Subtraction, not offset + size: untrusted ZIP64 values may overflow.
    return offset <= limit && size <= limit - offset;
}

static bool read_zip64_value(const std::vector<unsigned char>& extra, size_t& offset, size_t end, uint64_t& value, size_t width = 8)
{
    if (end > extra.size() || offset > end || width > end - offset)
        return false;

    value = 0;
    for (size_t i = 0; i < width; i++)
        value |= (uint64_t)extra[offset + i] << (8 * i);
    offset += width;
    return true;
}

static bool parse_zip64_extra(const std::vector<unsigned char>& extra, uint64_t& uncompressed_size, uint64_t& compressed_size, uint64_t* local_header_offset = 0, uint32_t* disk = 0)
{
    const bool need_uncompressed = uncompressed_size == 0xffffffffu;
    const bool need_compressed = compressed_size == 0xffffffffu;
    const bool need_offset = local_header_offset && *local_header_offset == 0xffffffffu;
    const bool need_disk = disk && *disk == 0xffff;
    bool found_zip64 = false;
    for (size_t offset = 0; offset < extra.size();)
    {
        if (extra.size() - offset < 4)
            return false;
        const uint16_t id = extra[offset] | ((uint16_t)extra[offset + 1] << 8);
        const uint16_t size = extra[offset + 2] | ((uint16_t)extra[offset + 3] << 8);
        offset += 4;
        if (size > extra.size() - offset)
            return false;
        const size_t end = offset + size;
        if (id == 0x0001)
        {
            if (found_zip64)
                return false;
            found_zip64 = true;
            if ((need_uncompressed && !read_zip64_value(extra, offset, end, uncompressed_size))
                || (need_compressed && !read_zip64_value(extra, offset, end, compressed_size))
                || (need_offset && !read_zip64_value(extra, offset, end, *local_header_offset)))
                return false;
            if (need_disk)
            {
                uint64_t value = 0;
                if (!read_zip64_value(extra, offset, end, value, 4))
                    return false;
                *disk = (uint32_t)value;
            }
            // Older pnnx writers include surplus offset/disk bytes in local
            // ZIP64 extras. Ignore surplus bytes INSIDE this bounded subfield.
        }
        offset = end;
    }

    return found_zip64 || !(need_uncompressed || need_compressed || need_offset || need_disk);
}

// Bound parser work and retained metadata independently of advertised payload
// sizes. Unknown compression is indexed, never decompressed or allocated here.
static const uint64_t ZIP_MAX_DIRECTORY_SIZE = 256ull * 1024 * 1024;
static const uint64_t ZIP_MAX_ENTRIES = 1024 * 1024;
static const uint64_t ZIP_MAX_NAME_BYTES = 64ull * 1024 * 1024;
static const uint16_t ZIP_ENCRYPTED_FLAGS = 0x0001 | 0x0040 | 0x2000;

int StoreZipReader::open(const std::string& path)
{
    close();

    fp = fopen(path.c_str(), "rb");
    if (!fp)
    {
        fprintf(stderr, "open failed\n");
        return -1;
    }

    int result = -1;
    try
    {
        result = open_archive();
    }
    catch (const std::bad_alloc&)
    {
        fprintf(stderr, "zip metadata allocation failed\n");
    }
    catch (const std::length_error&)
    {
        fprintf(stderr, "zip metadata length exceeds container limits\n");
    }
    if (result != 0)
    {
        fprintf(stderr, "invalid or unsupported zip archive\n");
        close();
    }
    return result;
}

int StoreZipReader::open_archive()
{
    if (file_seek(fp, 0, SEEK_END) != 0)
        return -1;

    const int64_t archive_size_i64 = file_tell(fp);
    if (archive_size_i64 < 22)
        return -1;
    archive_size = (uint64_t)archive_size_i64;

    const uint64_t tail_size = (std::min<uint64_t>)(archive_size, 65557);
    std::vector<unsigned char> tail((size_t)tail_size);
    if (file_seek(fp, archive_size_i64 - (int64_t)tail_size, SEEK_SET) != 0 || !read_bytes(fp, tail.data(), tail.size()))
        return -1;

    size_t eocd_offset = tail.size() - 22;
    bool found_eocd = false;
    while (true)
    {
        uint32_t signature = 0;
        memcpy(&signature, tail.data() + eocd_offset, sizeof(signature));
        if (signature == 0x06054b50)
        {
            end_of_central_directory_record candidate;
            memcpy(&candidate, tail.data() + eocd_offset + 4, sizeof(candidate));
            if (eocd_offset + 4 + sizeof(candidate) + candidate.comment_length == tail.size())
            {
                found_eocd = true;
                break;
            }
        }
        if (eocd_offset == 0)
            break;
        eocd_offset--;
    }
    if (!found_eocd)
        return -1;

    end_of_central_directory_record eocdr;
    memcpy(&eocdr, tail.data() + eocd_offset + 4, sizeof(eocdr));
    const uint64_t absolute_eocd_offset = archive_size - tail_size + eocd_offset;
    const bool need_zip64 = eocdr.disk_number == 0xffff || eocdr.start_disk == 0xffff
                           || eocdr.cd_records == 0xffff || eocdr.total_cd_records == 0xffff
                           || eocdr.cd_size == 0xffffffffu || eocdr.cd_offset == 0xffffffffu;
    uint64_t central_directory_offset = eocdr.cd_offset;
    uint64_t central_directory_size = eocdr.cd_size;
    uint64_t record_count = eocdr.total_cd_records;
    uint64_t directory_limit = absolute_eocd_offset;

    // A ZIP64 trailer may be present even when no legacy field is saturated.
    bool have_zip64 = false;
    zip64_end_of_central_directory_locator locator = {};
    const uint64_t locator_size = 4 + sizeof(locator);
    if (absolute_eocd_offset >= locator_size)
    {
        if (file_seek(fp, (int64_t)(absolute_eocd_offset - locator_size), SEEK_SET) != 0)
            return -1;
        uint32_t signature = 0;
        if (!read_bytes(fp, &signature, sizeof(signature)))
            return -1;
        if (signature == 0x07064b50)
        {
            if (!read_bytes(fp, &locator, sizeof(locator)))
                return -1;
            have_zip64 = true;
        }
    }
    if (need_zip64 && !have_zip64)
        return -1;

    if (have_zip64)
    {
        const uint64_t locator_offset = absolute_eocd_offset - locator_size;
        zip64_end_of_central_directory_record zip64_eocdr;
        if (locator.eocdr64_disk_number != 0 || locator.disk_count != 1
            || !zip_range_within(locator.eocdr64_offset, 4 + sizeof(zip64_eocdr), locator_offset)
            || file_seek(fp, (int64_t)locator.eocdr64_offset, SEEK_SET) != 0)
            return -1;
        uint32_t signature = 0;
        if (!read_bytes(fp, &signature, sizeof(signature)) || signature != 0x06064b50
            || !read_bytes(fp, &zip64_eocdr, sizeof(zip64_eocdr)))
            return -1;

        // The fixed record and its extensible data must end at the locator.
        if (zip64_eocdr.size_of_eocd64_m12 < 44
            || zip64_eocdr.size_of_eocd64_m12 != locator_offset - locator.eocdr64_offset - 12
            || zip64_eocdr.disk_number != 0 || zip64_eocdr.start_disk != 0
            || zip64_eocdr.cd_records != zip64_eocdr.total_cd_records)
            return -1;
        if ((eocdr.disk_number != 0xffff && eocdr.disk_number != zip64_eocdr.disk_number)
            || (eocdr.start_disk != 0xffff && eocdr.start_disk != zip64_eocdr.start_disk)
            || (eocdr.cd_records != 0xffff && eocdr.cd_records != zip64_eocdr.cd_records)
            || (eocdr.total_cd_records != 0xffff && eocdr.total_cd_records != zip64_eocdr.total_cd_records)
            || (eocdr.cd_size != 0xffffffffu && eocdr.cd_size != zip64_eocdr.cd_size)
            || (eocdr.cd_offset != 0xffffffffu && eocdr.cd_offset != zip64_eocdr.cd_offset))
            return -1;
        central_directory_offset = zip64_eocdr.cd_offset;
        central_directory_size = zip64_eocdr.cd_size;
        record_count = zip64_eocdr.total_cd_records;
        directory_limit = locator.eocdr64_offset;
    }
    else if (eocdr.disk_number != 0 || eocdr.start_disk != 0 || eocdr.cd_records != eocdr.total_cd_records)
    {
        return -1;
    }

    if (!zip_range_within(central_directory_offset, central_directory_size, directory_limit)
        || record_count > central_directory_size / 46
        || record_count > ZIP_MAX_ENTRIES || central_directory_size > ZIP_MAX_DIRECTORY_SIZE)
        return -1;

    const uint64_t directory_end = central_directory_offset + central_directory_size;
    uint64_t cursor = central_directory_offset;
    uint64_t name_bytes = 0;
    for (uint64_t record_index = 0; record_index < record_count; record_index++)
    {
        uint32_t signature = 0;
        central_directory_file_header cdfh;
        if (!zip_range_within(cursor, 4 + sizeof(cdfh), directory_end)
            || file_seek(fp, (int64_t)cursor, SEEK_SET) != 0
            || !read_bytes(fp, &signature, sizeof(signature)) || signature != 0x02014b50
            || !read_bytes(fp, &cdfh, sizeof(cdfh)))
            return -1;
        cursor += 4 + sizeof(cdfh);
        const uint64_t variable_size = (uint64_t)cdfh.file_name_length + cdfh.extra_field_length + cdfh.file_comment_length;
        if (!zip_range_within(cursor, variable_size, directory_end)
            || (cdfh.flag & ZIP_ENCRYPTED_FLAGS)
            || cdfh.file_name_length > ZIP_MAX_NAME_BYTES - name_bytes)
            return -1;
        name_bytes += cdfh.file_name_length;
        cursor += variable_size;

        std::string name(cdfh.file_name_length, '\0');
        std::vector<unsigned char> extra(cdfh.extra_field_length);
        if (name.empty() || !read_bytes(fp, &name[0], name.size()) || !read_bytes(fp, extra.data(), extra.size())
            || name.find('\0') != std::string::npos || filemetas.find(name) != filemetas.end())
            return -1;

        uint64_t compressed_size = cdfh.compressed_size;
        uint64_t uncompressed_size = cdfh.uncompressed_size;
        uint64_t local_header_offset = cdfh.lfh_offset;
        uint32_t disk = cdfh.start_disk;
        if (!parse_zip64_extra(extra, uncompressed_size, compressed_size, &local_header_offset, &disk)
            || disk != 0 || (cdfh.compression == 0 && compressed_size != uncompressed_size))
            return -1;

        uint32_t local_signature = 0;
        local_file_header lfh;
        if (!zip_range_within(local_header_offset, 4 + sizeof(lfh), central_directory_offset)
            || file_seek(fp, (int64_t)local_header_offset, SEEK_SET) != 0
            || !read_bytes(fp, &local_signature, sizeof(local_signature)) || local_signature != 0x04034b50
            || !read_bytes(fp, &lfh, sizeof(lfh)))
            return -1;
        if (lfh.compression != cdfh.compression || lfh.flag != cdfh.flag
            || (lfh.flag & ZIP_ENCRYPTED_FLAGS) || lfh.file_name_length != cdfh.file_name_length)
            return -1;

        const uint64_t local_variable_offset = local_header_offset + 4 + sizeof(lfh);
        const uint64_t local_variable_size = (uint64_t)lfh.file_name_length + lfh.extra_field_length;
        if (!zip_range_within(local_variable_offset, local_variable_size, central_directory_offset))
            return -1;
        const uint64_t data_offset = local_variable_offset + local_variable_size;
        if (!zip_range_within(data_offset, compressed_size, central_directory_offset))
            return -1;

        std::string local_name(lfh.file_name_length, '\0');
        std::vector<unsigned char> local_extra(lfh.extra_field_length);
        if (!read_bytes(fp, &local_name[0], local_name.size()) || local_name != name
            || !read_bytes(fp, local_extra.data(), local_extra.size()))
            return -1;
        uint64_t local_compressed_size = lfh.compressed_size;
        uint64_t local_uncompressed_size = lfh.uncompressed_size;
        if (!parse_zip64_extra(local_extra, local_uncompressed_size, local_compressed_size))
            return -1;

        if (cdfh.flag & 8)
        {
            // Descriptor-mode headers may use zero placeholders (also inside
            // ZIP64 extras), but any nonzero local value must still agree.
            if ((lfh.crc32 != 0 && lfh.crc32 != cdfh.crc32)
                || (local_compressed_size != 0 && local_compressed_size != compressed_size)
                || (local_uncompressed_size != 0 && local_uncompressed_size != uncompressed_size))
                return -1;
        }
        else if (lfh.crc32 != cdfh.crc32 || local_compressed_size != compressed_size || local_uncompressed_size != uncompressed_size)
        {
            return -1;
        }

        StoreZipMeta meta;
        meta.offset = data_offset;
        meta.size = uncompressed_size;
        meta.crc32 = cdfh.crc32;
        meta.flags = cdfh.flag;
        meta.compression = cdfh.compression;
        if (!filemetas.insert(std::make_pair(name, meta)).second)
            return -1;
    }

    // Do not let a false count hide additional entries or trailing CD bytes.
    return cursor == directory_end ? 0 : -1;
}

std::vector<std::string> StoreZipReader::get_names() const
{
    std::vector<std::string> names;
    for (std::map<std::string, StoreZipMeta>::const_iterator it = filemetas.begin(); it != filemetas.end(); ++it)
    {
        names.push_back(it->first);
    }

    return names;
}

int StoreZipReader::get_file_compression(const std::string& name) const
{
    const std::map<std::string, StoreZipMeta>::const_iterator it = filemetas.find(name);
    return it == filemetas.end() ? -1 : it->second.compression;
}

int StoreZipReader::get_file_flags(const std::string& name) const
{
    const std::map<std::string, StoreZipMeta>::const_iterator it = filemetas.find(name);
    return it == filemetas.end() ? -1 : it->second.flags;
}

uint64_t StoreZipReader::get_file_size(const std::string& name) const
{
    if (filemetas.find(name) == filemetas.end())
    {
        fprintf(stderr, "no such file %s\n", name.c_str());
        return 0;
    }

    return filemetas.at(name).size;
}

uint64_t StoreZipReader::get_archive_size() const
{
    return archive_size;
}

int StoreZipReader::read_file(const std::string& name, char* data)
{
    const std::map<std::string, StoreZipMeta>::const_iterator it = filemetas.find(name);
    if (!fp || it == filemetas.end())
    {
        fprintf(stderr, "no such file %s\n", name.c_str());
        return -1;
    }

    const StoreZipMeta& meta = it->second;
    if (meta.compression != 0 || (meta.flags & ZIP_ENCRYPTED_FLAGS))
    {
        fprintf(stderr, "compressed or encrypted zip record is not supported %s\n", name.c_str());
        return -1;
    }
    if (meta.size > (std::numeric_limits<size_t>::max)() || (meta.size != 0 && !data)
        || !zip_range_within(meta.offset, meta.size, (uint64_t)(std::numeric_limits<int64_t>::max)()))
    {
        fprintf(stderr, "invalid zip read buffer or size %s\n", name.c_str());
        return -1;
    }

    if (file_seek(fp, (int64_t)meta.offset, SEEK_SET) != 0)
    {
        fprintf(stderr, "seek failed %s\n", name.c_str());
        return -1;
    }

    if (!read_bytes(fp, data, (size_t)meta.size))
    {
        fprintf(stderr, "read failed %s\n", name.c_str());
        return -1;
    }

    // Zero is a real CRC, NOT an opt-out; empty STORE records are checked too.
    if (CRC32_buffer((const unsigned char*)data, (size_t)meta.size) != meta.crc32)
    {
        fprintf(stderr, "zip CRC mismatch %s\n", name.c_str());
        return -1;
    }

    return 0;
}

int StoreZipReader::close()
{
    const int result = fp && fclose(fp) != 0 ? -1 : 0;
    fp = 0;
    archive_size = 0;
    filemetas.clear();
    return result;
}

StoreZipWriter::StoreZipWriter()
{
    fp = 0;
    failed = false;
}

StoreZipWriter::~StoreZipWriter()
{
    close();
}

int StoreZipWriter::open(const std::string& path)
{
    if (close() != 0)
        return -1;

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
    if (!fp || failed || name.empty() || name.size() > 65535 || name.find('\0') != std::string::npos
        || size > (std::numeric_limits<size_t>::max)() || (size != 0 && !data))
        return -1;
    for (const StoreZipMeta& meta : filemetas)
    {
        if (meta.name == name)
            return -1;
    }

    const int64_t offset = file_tell(fp);
    if (offset < 0)
    {
        failed = true;
        return -1;
    }
    const uint64_t header_size = 4 + sizeof(local_file_header) + name.size() + 4 + sizeof(zip64_extended_extra_field);
    const uint64_t file_limit = (uint64_t)(std::numeric_limits<int64_t>::max)();
    if (!zip_range_within((uint64_t)offset, header_size, file_limit)
        || !zip_range_within((uint64_t)offset + header_size, size, file_limit))
        return -1;

    // Commit all potentially allocating metadata BEFORE any file bytes. An
    // allocation failure leaves the writer usable and no partial local header.
    try
    {
        StoreZipMeta szm;
        szm.name = name;
        szm.lfh_offset = (uint64_t)offset;
        szm.crc32 = CRC32_buffer((const unsigned char*)data, (size_t)size);
        szm.size = size;
        filemetas.push_back(szm);
    }
    catch (const std::bad_alloc&)
    {
        fprintf(stderr, "zip writer metadata allocation failed\n");
        return -1;
    }
    catch (const std::length_error&)
    {
        fprintf(stderr, "zip writer metadata length exceeds container limits\n");
        return -1;
    }

    const uint32_t signature = 0x04034b50;
    local_file_header lfh = {};
    lfh.version = 45;
    lfh.crc32 = filemetas.back().crc32;
    lfh.compressed_size = 0xffffffffu;
    lfh.uncompressed_size = 0xffffffffu;
    lfh.file_name_length = (uint16_t)name.size();

    zip64_extended_extra_field zip64_eef = {};
    zip64_eef.uncompressed_size = size;
    zip64_eef.compressed_size = size;
    const uint16_t extra_id = 0x0001;
    const uint16_t extra_size = sizeof(zip64_eef);
    lfh.extra_field_length = sizeof(extra_id) + sizeof(extra_size) + sizeof(zip64_eef);

    if (!write_bytes(fp, &signature, sizeof(signature))
        || !write_bytes(fp, &lfh, sizeof(lfh))
        || !write_bytes(fp, name.data(), name.size())
        || !write_bytes(fp, &extra_id, sizeof(extra_id))
        || !write_bytes(fp, &extra_size, sizeof(extra_size))
        || !write_bytes(fp, &zip64_eef, sizeof(zip64_eef))
        || !write_bytes(fp, data, (size_t)size))
    {
        failed = true;
        return -1;
    }
    return 0;
}

int StoreZipWriter::write_central_directory()
{
    const int64_t offset = file_tell(fp);
    if (offset < 0)
        return -1;

    // Prove the final CD and ZIP64 trailer offsets fit the signed 64-bit I/O
    // API before emitting anything; this loop does not allocate.
    const uint64_t file_limit = (uint64_t)(std::numeric_limits<int64_t>::max)();
    uint64_t end = (uint64_t)offset;
    for (const StoreZipMeta& szm : filemetas)
    {
        const uint64_t record_size = 4 + sizeof(central_directory_file_header) + szm.name.size() + 4 + sizeof(zip64_extended_extra_field);
        if (!zip_range_within(end, record_size, file_limit))
            return -1;
        end += record_size;
    }
    const uint64_t trailer_size = 12 + sizeof(zip64_end_of_central_directory_record)
                                  + sizeof(zip64_end_of_central_directory_locator) + sizeof(end_of_central_directory_record);
    if (!zip_range_within(end, trailer_size, file_limit))
        return -1;

    for (const StoreZipMeta& szm : filemetas)
    {
        const uint32_t signature = 0x02014b50;
        central_directory_file_header cdfh = {};
        cdfh.version_made = 45;
        cdfh.version = 45;
        cdfh.crc32 = szm.crc32;
        cdfh.compressed_size = 0xffffffffu;
        cdfh.uncompressed_size = 0xffffffffu;
        cdfh.file_name_length = (uint16_t)szm.name.size();
        cdfh.start_disk = 0xffff;
        cdfh.lfh_offset = 0xffffffffu;

        zip64_extended_extra_field zip64_eef = {};
        zip64_eef.uncompressed_size = szm.size;
        zip64_eef.compressed_size = szm.size;
        zip64_eef.lfh_offset = szm.lfh_offset;
        const uint16_t extra_id = 0x0001;
        const uint16_t extra_size = sizeof(zip64_eef);
        cdfh.extra_field_length = sizeof(extra_id) + sizeof(extra_size) + sizeof(zip64_eef);

        if (!write_bytes(fp, &signature, sizeof(signature))
            || !write_bytes(fp, &cdfh, sizeof(cdfh))
            || !write_bytes(fp, szm.name.data(), szm.name.size())
            || !write_bytes(fp, &extra_id, sizeof(extra_id))
            || !write_bytes(fp, &extra_size, sizeof(extra_size))
            || !write_bytes(fp, &zip64_eef, sizeof(zip64_eef)))
            return -1;
    }

    const int64_t offset2 = file_tell(fp);
    if (offset2 < offset || (uint64_t)offset2 != end)
        return -1;

    {
        const uint32_t signature = 0x06064b50;
        zip64_end_of_central_directory_record eocdr64 = {};
        eocdr64.size_of_eocd64_m12 = sizeof(eocdr64) - 8;
        eocdr64.version_made_by = 45;
        eocdr64.version_min_required = 45;
        eocdr64.cd_records = filemetas.size();
        eocdr64.total_cd_records = filemetas.size();
        eocdr64.cd_size = offset2 - offset;
        eocdr64.cd_offset = offset;
        if (!write_bytes(fp, &signature, sizeof(signature)) || !write_bytes(fp, &eocdr64, sizeof(eocdr64)))
            return -1;
    }

    {
        const uint32_t signature = 0x07064b50;
        zip64_end_of_central_directory_locator eocdl64 = {};
        eocdl64.eocdr64_disk_number = 0;
        eocdl64.eocdr64_offset = offset2;
        eocdl64.disk_count = 1;
        if (!write_bytes(fp, &signature, sizeof(signature)) || !write_bytes(fp, &eocdl64, sizeof(eocdl64)))
            return -1;
    }

    {
        const uint32_t signature = 0x06054b50;
        end_of_central_directory_record eocdr = {};
        eocdr.cd_records = 0xffff;
        eocdr.total_cd_records = 0xffff;
        eocdr.cd_size = 0xffffffffu;
        eocdr.cd_offset = 0xffffffffu;
        if (!write_bytes(fp, &signature, sizeof(signature)) || !write_bytes(fp, &eocdr, sizeof(eocdr)))
            return -1;
    }

    return 0;
}

int StoreZipWriter::close()
{
    int result = 0;
    if (fp)
    {
        result = failed ? -1 : write_central_directory();
        // fclose also reports buffered write/flush failures.
        if (fclose(fp) != 0)
            result = -1;
    }
    fp = 0;
    failed = false;
    filemetas.clear();
    return result;
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

        std::vector<std::string> names = sz.get_names();

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
