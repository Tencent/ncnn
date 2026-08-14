// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "storezip.h"

#include <limits.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <map>
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

static uint32_t CRC32_TABLE[256];

static void CRC32_TABLE_INIT()
{
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
        CRC32_TABLE[i] = c;
    }
}

static uint32_t CRC32(uint32_t x, unsigned char ch)
{
    return (x >> 8) ^ CRC32_TABLE[(x ^ ch) & 0xff];
}

static uint32_t CRC32_buffer(const unsigned char* data, uint64_t len)
{
    uint32_t x = 0xffffffff;

    for (uint64_t i = 0; i < len; i++)
        x = CRC32(x, data[i]);

    return x ^ 0xffffffff;
}

StoreZipReader::StoreZipReader()
{
    fp = 0;
    memory = 0;
    memory_size = 0;
    memory_position = 0;
    data_limit = 0;

    CRC32_TABLE_INIT();
}

StoreZipReader::~StoreZipReader()
{
    close();
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

static bool checked_add_u64(uint64_t a, uint64_t b, uint64_t& result)
{
    if (a > UINT64_MAX - b)
        return false;

    result = a + b;
    return true;
}

static int file_seek(FILE* fp, uint64_t offset, int origin)
{
    if (offset > INT64_MAX)
        return -1;

#if defined(_WIN32)
    return _fseeki64(fp, (__int64)offset, origin);
#else
    const off_t off = (off_t)offset;
    if (off < 0 || (uint64_t)off != offset)
        return -1;
    return fseeko(fp, off, origin);
#endif
}

static int64_t file_tell(FILE* fp)
{
#if defined(_WIN32)
    return _ftelli64(fp);
#else
    return (int64_t)ftello(fp);
#endif
}

static bool normalize_zip_name(const std::string& name, std::string& normalized)
{
    normalized.clear();

    if (name.empty() || name.find('\0') != std::string::npos || name[0] == '/' || name[0] == '\\' ||
        (name.size() >= 2 && name[1] == ':') || name.find('\\') != std::string::npos)
        return false;

    size_t begin = 0;
    while (begin <= name.size())
    {
        const size_t end = name.find('/', begin);
        const std::string component = name.substr(begin, end == std::string::npos ? std::string::npos : end - begin);

        if (component == "..")
            return false;
        if (!component.empty() && component != ".")
        {
            if (!normalized.empty())
                normalized += '/';
            normalized += component;
        }

        if (end == std::string::npos)
            break;
        begin = end + 1;
    }

    return !normalized.empty();
}

static bool parse_zip64_extra(const std::vector<unsigned char>& extra,
                              bool need_uncompressed_size, bool need_compressed_size,
                              bool need_lfh_offset, bool need_start_disk,
                              uint64_t& uncompressed_size, uint64_t& compressed_size,
                              uint64_t& lfh_offset, uint32_t& start_disk)
{
    size_t offset = 0;
    while (offset + 4 <= extra.size())
    {
        const uint16_t id = read_le16(&extra[offset]);
        const uint16_t size = read_le16(&extra[offset + 2]);
        offset += 4;
        if (size > extra.size() - offset)
            return false;

        if (id == 0x0001)
        {
            size_t p = offset;
            const size_t end = offset + size;

            if (need_uncompressed_size)
            {
                if (end - p < 8)
                    return false;
                uncompressed_size = read_le64(&extra[p]);
                p += 8;
            }
            if (need_compressed_size)
            {
                if (end - p < 8)
                    return false;
                compressed_size = read_le64(&extra[p]);
                p += 8;
            }
            if (need_lfh_offset)
            {
                if (end - p < 8)
                    return false;
                lfh_offset = read_le64(&extra[p]);
                p += 8;
            }
            if (need_start_disk)
            {
                if (end - p < 4)
                    return false;
                start_disk = read_le32(&extra[p]);
            }

            return true;
        }

        offset += size;
    }

    return !(need_uncompressed_size || need_compressed_size || need_lfh_offset || need_start_disk);
}

int StoreZipReader::fail(const std::string& message)
{
    close();
    error = message;
    fprintf(stderr, "storezip: %s\n", message.c_str());
    return -1;
}

int StoreZipReader::open(const std::string& path)
{
    close();
    error.clear();

    fp = fopen(path.c_str(), "rb");
    if (!fp)
        return fail("open failed " + path);

    return parse();
}

int StoreZipReader::open(const unsigned char* data, size_t size)
{
    close();
    error.clear();

    if (!data && size != 0)
        return fail("invalid memory archive");

    memory = data;
    memory_size = size;
    return parse();
}

int StoreZipReader::seek(uint64_t offset, int origin)
{
    if (fp)
        return file_seek(fp, offset, origin);

    uint64_t base;
    if (origin == SEEK_SET)
        base = 0;
    else if (origin == SEEK_CUR)
        base = memory_position;
    else if (origin == SEEK_END)
        base = memory_size;
    else
        return -1;

    if (base > memory_size || offset > memory_size - base)
        return -1;
    memory_position = base + offset;
    return 0;
}

int64_t StoreZipReader::tell() const
{
    if (fp)
        return file_tell(fp);
    if (memory_position > INT64_MAX)
        return -1;
    return (int64_t)memory_position;
}

bool StoreZipReader::read(void* data, size_t size)
{
    if (fp)
        return size == 0 || fread(data, 1, size, fp) == size;
    if (memory_position > memory_size || size > memory_size - memory_position)
        return false;
    if (size)
        memcpy(data, memory + memory_position, size);
    memory_position += size;
    return true;
}

int StoreZipReader::parse()
{
    if (!fp && !memory)
        return fail("archive is not open");

    if (seek(0, SEEK_END) != 0)
        return fail("failed to seek to end of archive");

    const int64_t file_size_signed = tell();
    if (file_size_signed < 0)
        return fail("failed to determine archive size");
    const uint64_t archive_size = (uint64_t)file_size_signed;

    if (archive_size < 22)
        return fail("truncated archive: end of central directory is missing");

    const uint64_t search_size_u64 = std::min<uint64_t>(archive_size, 22u + 65535u);
    const size_t search_size = (size_t)search_size_u64;
    std::vector<unsigned char> tail(search_size);
    if (seek(archive_size - search_size_u64, SEEK_SET) != 0 || !read(&tail[0], tail.size()))
        return fail("failed to read end of archive");

    size_t eocd_tail_offset = SIZE_MAX;
    for (size_t i = search_size - 22 + 1; i-- > 0;)
    {
        if (read_le32(&tail[i]) == 0x06054b50 && i + 22u + read_le16(&tail[i + 20]) == tail.size())
        {
            eocd_tail_offset = i;
            break;
        }
    }
    if (eocd_tail_offset == SIZE_MAX)
        return fail("invalid archive: end of central directory was not found");

    const unsigned char* eocd = &tail[eocd_tail_offset];
    uint32_t disk_number = read_le16(eocd + 4);
    uint32_t start_disk = read_le16(eocd + 6);
    uint64_t cd_records = read_le16(eocd + 8);
    uint64_t total_cd_records = read_le16(eocd + 10);
    uint64_t cd_size = read_le32(eocd + 12);
    uint64_t cd_offset = read_le32(eocd + 16);
    const uint64_t eocd_offset = archive_size - search_size_u64 + eocd_tail_offset;
    uint64_t central_directory_limit = eocd_offset;

    const bool needs_zip64 = disk_number == 0xffffu || start_disk == 0xffffu ||
                             cd_records == 0xffffu || total_cd_records == 0xffffu ||
                             cd_size == 0xffffffffu || cd_offset == 0xffffffffu;
    if (needs_zip64)
    {
        if (eocd_offset < 20)
            return fail("invalid ZIP64 archive: locator is missing");

        unsigned char locator[20];
        if (seek(eocd_offset - 20, SEEK_SET) != 0 || !read(locator, sizeof(locator)))
            return fail("truncated ZIP64 locator");
        if (read_le32(locator) != 0x07064b50)
            return fail("invalid ZIP64 archive: locator signature is missing");
        if (read_le32(locator + 4) != 0 || read_le32(locator + 16) != 1)
            return fail("multi-disk ZIP64 archives are not supported");

        const uint64_t zip64_eocd_offset = read_le64(locator + 8);
        unsigned char zip64_eocd[56];
        if (zip64_eocd_offset > eocd_offset - 20 || seek(zip64_eocd_offset, SEEK_SET) != 0 || !read(zip64_eocd, sizeof(zip64_eocd)))
            return fail("truncated ZIP64 end of central directory");
        if (read_le32(zip64_eocd) != 0x06064b50 || read_le64(zip64_eocd + 4) < 44)
            return fail("invalid ZIP64 end of central directory");

        uint64_t zip64_eocd_total_size = 0;
        uint64_t zip64_eocd_end = 0;
        if (!checked_add_u64(read_le64(zip64_eocd + 4), 12u, zip64_eocd_total_size) ||
            !checked_add_u64(zip64_eocd_offset, zip64_eocd_total_size, zip64_eocd_end) ||
            zip64_eocd_end > eocd_offset - 20)
            return fail("ZIP64 end of central directory is out of bounds");

        disk_number = read_le32(zip64_eocd + 16);
        start_disk = read_le32(zip64_eocd + 20);
        cd_records = read_le64(zip64_eocd + 24);
        total_cd_records = read_le64(zip64_eocd + 32);
        cd_size = read_le64(zip64_eocd + 40);
        cd_offset = read_le64(zip64_eocd + 48);
        central_directory_limit = zip64_eocd_offset;
    }

    if (disk_number != 0 || start_disk != 0 || cd_records != total_cd_records)
        return fail("multi-disk ZIP archives are not supported");

    uint64_t cd_end = 0;
    if (!checked_add_u64(cd_offset, cd_size, cd_end) || cd_end > central_directory_limit || cd_end > archive_size)
        return fail("central directory is outside the archive");
    if (cd_records > cd_size / 46u + (cd_size == 0 ? 0u : 1u))
        return fail("central directory record count is inconsistent with its size");
    if (seek(cd_offset, SEEK_SET) != 0)
        return fail("failed to seek to central directory");
    data_limit = cd_offset;

    uint64_t central_position = cd_offset;
    for (uint64_t record_index = 0; record_index < total_cd_records; record_index++)
    {
        if (central_position > cd_end || cd_end - central_position < 46)
            return fail("truncated central directory record");

        unsigned char central[46];
        if (!read(central, sizeof(central)) || read_le32(central) != 0x02014b50)
            return fail("invalid central directory record signature");

        const uint16_t flag = read_le16(central + 8);
        const uint16_t compression = read_le16(central + 10);
        const uint32_t crc32 = read_le32(central + 16);
        uint64_t compressed_size = read_le32(central + 20);
        uint64_t uncompressed_size = read_le32(central + 24);
        const uint16_t file_name_length = read_le16(central + 28);
        const uint16_t extra_field_length = read_le16(central + 30);
        const uint16_t file_comment_length = read_le16(central + 32);
        uint32_t entry_start_disk = read_le16(central + 34);
        uint64_t lfh_offset = read_le32(central + 42);

        uint64_t variable_size = (uint64_t)file_name_length + extra_field_length + file_comment_length;
        uint64_t record_end = 0;
        if (!checked_add_u64(central_position + 46u, variable_size, record_end) || record_end > cd_end)
            return fail("central directory variable fields are out of bounds");

        std::string raw_name(file_name_length, '\0');
        std::vector<unsigned char> extra(extra_field_length);
        if (!read(raw_name.empty() ? 0 : &raw_name[0], raw_name.size()) ||
            !read(extra.empty() ? 0 : &extra[0], extra.size()) ||
            (file_comment_length && seek(file_comment_length, SEEK_CUR) != 0))
            return fail("truncated central directory variable fields");
        central_position = record_end;

        const bool need_uncompressed_size = uncompressed_size == 0xffffffffu;
        const bool need_compressed_size = compressed_size == 0xffffffffu;
        const bool need_lfh_offset = lfh_offset == 0xffffffffu;
        const bool need_start_disk = entry_start_disk == 0xffffu;
        if (!parse_zip64_extra(extra, need_uncompressed_size, need_compressed_size, need_lfh_offset, need_start_disk,
                               uncompressed_size, compressed_size, lfh_offset, entry_start_disk))
            return fail("missing or malformed ZIP64 extended information for " + raw_name);

        if (entry_start_disk != 0)
            return fail("multi-disk record is not supported: " + raw_name);
        if (flag & 0x0001)
            return fail("encrypted record is not supported: " + raw_name);
        if (flag & (uint16_t)~0x0808u)
            return fail("unsupported ZIP general-purpose flag for " + raw_name);
        if (compression != 0)
            return fail("unsupported compression method for " + raw_name);
        if (compressed_size != uncompressed_size)
            return fail("stored record has mismatched compressed and uncompressed sizes: " + raw_name);

        std::string name;
        if (!normalize_zip_name(raw_name, name))
            return fail("unsafe ZIP record name: " + raw_name);

        // Directory entries carry no model data and are not exposed as records.
        if (!raw_name.empty() && raw_name[raw_name.size() - 1] == '/')
            continue;

        if (lfh_offset > cd_offset || cd_offset - lfh_offset < 30)
            return fail("local header is outside the file-data area: " + name);

        StoreZipMeta fm;
        fm.offset = lfh_offset;
        fm.size = uncompressed_size;
        fm.crc32 = crc32;
        fm.flag = flag;
        if (!filemetas.insert(std::make_pair(name, fm)).second)
            return fail("duplicate normalized ZIP record name: " + name);
    }

    if (central_position != cd_end)
        return fail("central directory size does not match its records");

    return 0;
}

std::vector<std::string> StoreZipReader::get_names() const
{
    std::vector<std::string> names;
    names.reserve(filemetas.size());
    for (std::map<std::string, StoreZipMeta>::const_iterator it = filemetas.begin(); it != filemetas.end(); ++it)
    {
        names.push_back(it->first);
    }

    return names;
}

uint64_t StoreZipReader::get_file_size(const std::string& name) const
{
    std::map<std::string, StoreZipMeta>::const_iterator it = filemetas.find(name);
    if (it == filemetas.end())
    {
        fprintf(stderr, "no such file %s\n", name.c_str());
        return 0;
    }

    return it->second.size;
}

int StoreZipReader::read_file(const std::string& name, char* data)
{
    std::map<std::string, StoreZipMeta>::const_iterator it = filemetas.find(name);
    if (it == filemetas.end())
    {
        error = "no such file " + name;
        return -1;
    }

    return read_file(name, it->second, data);
}

int StoreZipReader::read_file(const std::string& name, const StoreZipMeta& meta, char* data)
{
    if (meta.size > SIZE_MAX || (meta.size != 0 && !data))
    {
        error = "invalid destination buffer for " + name;
        return -1;
    }

    unsigned char local[30];
    if ((!fp && !memory) || seek(meta.offset, SEEK_SET) != 0 || !read(local, sizeof(local)) || read_le32(local) != 0x04034b50)
    {
        error = "invalid local header for " + name;
        return -1;
    }

    const uint16_t local_name_length = read_le16(local + 26);
    const uint16_t local_extra_length = read_le16(local + 28);
    if (read_le16(local + 6) != meta.flag || read_le16(local + 8) != 0)
    {
        error = "local and central header metadata disagree for " + name;
        return -1;
    }

    uint64_t data_offset = 0;
    uint64_t data_end = 0;
    if (!checked_add_u64(meta.offset, 30u + (uint64_t)local_name_length + local_extra_length, data_offset) ||
        !checked_add_u64(data_offset, meta.size, data_end) || data_end > data_limit)
    {
        error = "record data is outside the file-data area: " + name;
        return -1;
    }

    std::string local_name(local_name_length, '\0');
    std::string normalized_local_name;
    if (!read(local_name.empty() ? 0 : &local_name[0], local_name.size()) ||
        !normalize_zip_name(local_name, normalized_local_name) || normalized_local_name != name ||
        seek(data_offset, SEEK_SET) != 0 || !read(data, (size_t)meta.size))
    {
        error = "failed to read record " + name;
        return -1;
    }
    if (CRC32_buffer((const unsigned char*)data, meta.size) != meta.crc32)
    {
        error = "CRC-32 mismatch for " + name;
        return -1;
    }

    return 0;
}

int StoreZipReader::read_file(const std::string& name, std::vector<unsigned char>& data)
{
    data.clear();
    std::map<std::string, StoreZipMeta>::const_iterator it = filemetas.find(name);
    if (it == filemetas.end())
    {
        error = "no such file " + name;
        return -1;
    }

    const uint64_t size = it->second.size;
    if (size > SIZE_MAX)
    {
        error = "record is too large for memory: " + name;
        return -1;
    }

    data.resize((size_t)size);
    return read_file(name, it->second, data.empty() ? 0 : (char*)&data[0]);
}

int StoreZipReader::close()
{
    if (fp)
    {
        fclose(fp);
        fp = 0;
    }
    memory = 0;
    memory_size = 0;
    memory_position = 0;
    filemetas.clear();
    data_limit = 0;

    return 0;
}

StoreZipWriter::StoreZipWriter()
{
    fp = 0;

    CRC32_TABLE_INIT();
}

StoreZipWriter::~StoreZipWriter()
{
    close();
}

int StoreZipWriter::open(const std::string& path)
{
    close();
    filemetas.clear();

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
    long offset = ftell(fp);

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

    long offset = ftell(fp);

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

    long offset2 = ftell(fp);

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
    filemetas.clear();

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
