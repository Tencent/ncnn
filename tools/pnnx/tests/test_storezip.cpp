// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Standalone C++11 test: compile this file together with ../src/storezip.cpp,
// with ../src on the include path. No Torch, zlib, or test framework is needed.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits>
#include <map>
#include <new>
#include <string>
#include <vector>

// Include standard headers first: expose only our classes for lifecycle checks.
#define private public
#include "storezip.h"
#undef private

// Deterministic allocation failure, never actual OOM. This executable is
// single-threaded and must not be combined with other tests' main functions.
static int allocation_budget = -1;
static size_t allocation_attempts = 0;
static size_t largest_allocation = 0;
static size_t oversized_allocation_attempts = 0;

void* operator new(size_t size)
{
    allocation_attempts++;
    if (size > largest_allocation)
        largest_allocation = size;
    // Keep malicious-count regression tests safe even if a future parser tries
    // to reserve untrusted counts. Every real fixture allocation is below 1 MiB.
    if (size > 1024 * 1024)
    {
        oversized_allocation_attempts++;
        throw std::bad_alloc();
    }
    if (allocation_budget == 0)
    {
        allocation_budget = -1;
        throw std::bad_alloc();
    }
    if (allocation_budget > 0)
        allocation_budget--;
    void* p = malloc(size ? size : 1);
    if (!p)
        throw std::bad_alloc();
    return p;
}

void* operator new[](size_t size)
{
    return ::operator new(size);
}

void operator delete(void* p)noexcept
{
    free(p);
}

void operator delete[](void* p) noexcept
{
    free(p);
}

#if defined(__cpp_sized_deallocation)
void operator delete(void* p, size_t)noexcept
{
    free(p);
}

void operator delete[](void* p, size_t) noexcept
{
    free(p);
}
#endif

typedef std::vector<unsigned char> Bytes;
static int test_failures = 0;

static void expect(bool condition, const char* message)
{
    if (!condition)
    {
        fprintf(stderr, "FAILED: %s\n", message);
        test_failures++;
    }
}

// Independent bitwise CRC oracle: no dependency on the implementation's table.
static uint32_t fixture_crc(const std::string& value)
{
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < value.size(); i++)
    {
        crc ^= (unsigned char)value[i];
        for (int bit = 0; bit < 8; bit++)
            crc = (crc >> 1) ^ ((crc & 1) ? 0xedb88320u : 0);
    }
    return crc ^ 0xffffffffu;
}

static void put(Bytes& bytes, size_t offset, uint64_t value, size_t width)
{
    // A fixture bug must not turn a negative parser test into an unsafe write.
    if (width > 8 || offset > bytes.size() || width > bytes.size() - offset)
        abort();
    for (size_t i = 0; i < width; i++)
        bytes[offset + i] = (unsigned char)(value >> (i * 8));
}

static void append_number(Bytes& bytes, uint64_t value, size_t width)
{
    const size_t offset = bytes.size();
    bytes.resize(offset + width);
    put(bytes, offset, value, width);
}

static void append(Bytes& bytes, const Bytes& value)
{
    bytes.insert(bytes.end(), value.begin(), value.end());
}

static void append(Bytes& bytes, const std::string& value)
{
    bytes.insert(bytes.end(), value.begin(), value.end());
}

static Bytes extra_field(uint16_t id, const Bytes& payload)
{
    if (payload.size() > 65531)
        abort();
    Bytes result;
    append_number(result, id, 2);
    append_number(result, payload.size(), 2);
    append(result, payload);
    return result;
}

enum Zip64EntryFields
{
    Zip64Uncompressed = 1,
    Zip64Compressed = 2,
    Zip64Offset = 4,
    Zip64Disk = 8
};

struct Entry
{
    Entry(const std::string& name_, const std::string& value_)
        : name(name_), value(value_), payload(value_), flags(0), method(0), zip64_fields(0), local_zip64_fields(0), disk(0), descriptor_signature(true), descriptor64(false)
    {
    }

    std::string name;
    std::string value;
    std::string payload;
    uint16_t flags;
    uint16_t method;
    unsigned int zip64_fields;
    unsigned int local_zip64_fields;
    uint32_t disk;
    bool descriptor_signature;
    bool descriptor64;
    Bytes local_extra;
    Bytes central_extra;
    std::string comment;
};

struct Fixture
{
    Bytes bytes;
    std::vector<size_t> local;
    std::vector<size_t> data;
    std::vector<size_t> central;
    std::vector<size_t> local_extra;
    std::vector<size_t> central_extra;
    size_t cd;
    size_t cd_size;
    size_t zip64;
    size_t locator;
    size_t eocd;
};

// All offsets and lengths are encoded explicitly in little endian. Mutations
// affect only this small byte vector; malicious advertised sizes never allocate.
static Fixture make_zip(const std::vector<Entry>& entries, bool zip64 = false, unsigned int saturated = 63, const std::string& comment = "")
{
    Fixture f = {};
    Bytes& b = f.bytes;
    for (size_t i = 0; i < entries.size(); i++)
    {
        const Entry& e = entries[i];
        const bool descriptor = (e.flags & 8) != 0;
        Bytes extra;
        if (e.local_zip64_fields)
        {
            Bytes payload;
            if (e.local_zip64_fields & Zip64Uncompressed)
                append_number(payload, descriptor ? 0 : e.value.size(), 8);
            if (e.local_zip64_fields & Zip64Compressed)
                append_number(payload, descriptor ? 0 : e.payload.size(), 8);
            extra = extra_field(1, payload);
        }
        append(extra, e.local_extra);
        if (e.name.size() > 65535 || extra.size() > 65535)
            abort();
        f.local.push_back(b.size());
        append_number(b, 0x04034b50, 4);
        append_number(b, e.local_zip64_fields ? 45 : 20, 2);
        append_number(b, e.flags, 2);
        append_number(b, e.method, 2);
        append_number(b, 0, 4); // time/date
        append_number(b, descriptor ? 0 : fixture_crc(e.value), 4);
        append_number(b, (e.local_zip64_fields & Zip64Compressed) ? 0xffffffffu : (descriptor ? 0 : e.payload.size()), 4);
        append_number(b, (e.local_zip64_fields & Zip64Uncompressed) ? 0xffffffffu : (descriptor ? 0 : e.value.size()), 4);
        append_number(b, e.name.size(), 2);
        append_number(b, extra.size(), 2);
        append(b, e.name);
        f.local_extra.push_back(b.size());
        append(b, extra);
        f.data.push_back(b.size());
        append(b, e.payload);
        if (descriptor)
        {
            if (e.descriptor_signature)
                append_number(b, 0x08074b50, 4);
            append_number(b, fixture_crc(e.value), 4);
            append_number(b, e.payload.size(), e.descriptor64 ? 8 : 4);
            append_number(b, e.value.size(), e.descriptor64 ? 8 : 4);
        }
    }

    f.cd = b.size();
    for (size_t i = 0; i < entries.size(); i++)
    {
        const Entry& e = entries[i];
        Bytes extra;
        if (e.zip64_fields)
        {
            Bytes payload;
            if (e.zip64_fields & Zip64Uncompressed)
                append_number(payload, e.value.size(), 8);
            if (e.zip64_fields & Zip64Compressed)
                append_number(payload, e.payload.size(), 8);
            if (e.zip64_fields & Zip64Offset)
                append_number(payload, f.local[i], 8);
            if (e.zip64_fields & Zip64Disk)
                append_number(payload, e.disk, 4);
            extra = extra_field(1, payload);
        }
        append(extra, e.central_extra);
        if (extra.size() > 65535 || e.comment.size() > 65535)
            abort();
        f.central.push_back(b.size());
        append_number(b, 0x02014b50, 4);
        append_number(b, 45, 2);
        append_number(b, 45, 2);
        append_number(b, e.flags, 2);
        append_number(b, e.method, 2);
        append_number(b, 0, 4); // time/date
        append_number(b, fixture_crc(e.value), 4);
        append_number(b, (e.zip64_fields & Zip64Compressed) ? 0xffffffffu : e.payload.size(), 4);
        append_number(b, (e.zip64_fields & Zip64Uncompressed) ? 0xffffffffu : e.value.size(), 4);
        append_number(b, e.name.size(), 2);
        append_number(b, extra.size(), 2);
        append_number(b, e.comment.size(), 2);
        append_number(b, (e.zip64_fields & Zip64Disk) ? 0xffff : e.disk, 2);
        append_number(b, 0, 2); // internal attributes
        append_number(b, 0, 4); // external attributes
        append_number(b, (e.zip64_fields & Zip64Offset) ? 0xffffffffu : f.local[i], 4);
        append(b, e.name);
        f.central_extra.push_back(b.size());
        append(b, extra);
        append(b, e.comment);
    }
    f.cd_size = b.size() - f.cd;
    if (zip64)
    {
        f.zip64 = b.size();
        append_number(b, 0x06064b50, 4);
        append_number(b, 44, 8);
        append_number(b, 45, 2);
        append_number(b, 45, 2);
        append_number(b, 0, 4);
        append_number(b, 0, 4);
        append_number(b, entries.size(), 8);
        append_number(b, entries.size(), 8);
        append_number(b, f.cd_size, 8);
        append_number(b, f.cd, 8);
        f.locator = b.size();
        append_number(b, 0x07064b50, 4);
        append_number(b, 0, 4);
        append_number(b, f.zip64, 8);
        append_number(b, 1, 4);
    }
    else
    {
        saturated = 0;
    }
    if (entries.size() >= 65535 || comment.size() > 65535)
        abort();
    f.eocd = b.size();
    append_number(b, 0x06054b50, 4);
    append_number(b, (saturated & 1) ? 0xffff : 0, 2);
    append_number(b, (saturated & 2) ? 0xffff : 0, 2);
    append_number(b, (saturated & 4) ? 0xffff : entries.size(), 2);
    append_number(b, (saturated & 8) ? 0xffff : entries.size(), 2);
    append_number(b, (saturated & 16) ? 0xffffffffu : f.cd_size, 4);
    append_number(b, (saturated & 32) ? 0xffffffffu : f.cd, 4);
    append_number(b, comment.size(), 2);
    append(b, comment);
    return f;
}

struct FixtureFile
{
    explicit FixtureFile(const char* path_ = "test_storezip_fixture.zip")
        : path(path_)
    {
    }
    ~FixtureFile()
    {
        remove(path.c_str());
    }

    bool save(const Fixture& f) const
    {
        FILE* fp = fopen(path.c_str(), "wb");
        if (!fp)
        {
            expect(false, "create fixture");
            return false;
        }
        bool ok = f.bytes.empty() || fwrite(f.bytes.data(), 1, f.bytes.size(), fp) == f.bytes.size();
        if (fclose(fp) != 0)
            ok = false;
        expect(ok, "write fixture");
        return ok;
    }

    std::string path;
};

static void expect_clean(const pnnx::StoreZipReader& reader)
{
    expect(reader.fp == 0, "reader file is closed");
    expect(reader.filemetas.empty() && reader.get_names().empty(), "reader metadata is cleared");
}

static void expect_contents(pnnx::StoreZipReader& reader, const std::vector<Entry>& entries)
{
    expect(reader.get_names().size() == entries.size(), "entry count");
    for (size_t i = 0; i < entries.size(); i++)
    {
        const Entry& e = entries[i];
        expect(reader.get_file_size(e.name) == e.value.size(), "entry size");
        std::vector<char> data(e.value.size());
        const int result = reader.read_file(e.name, data.empty() ? 0 : data.data());
        expect(result == 0, "consume STORE entry and verify CRC");
        expect(std::string(data.begin(), data.end()) == e.value, "entry contents");
    }
}

static void expect_valid(const Fixture& f, const std::vector<Entry>& entries, const char* message)
{
    FixtureFile file;
    if (!file.save(f))
        return;
    pnnx::StoreZipReader reader;
    const int result = reader.open(file.path);
    expect(result == 0, message);
    if (result == 0)
        expect_contents(reader, entries);
    expect(reader.close() == 0 && reader.close() == 0, "idempotent reader close");
    expect_clean(reader);
}

static void expect_rejected(const Fixture& f, const char* message)
{
    FixtureFile file;
    if (!file.save(f))
        return;
    pnnx::StoreZipReader reader;
    expect(reader.open(file.path) == -1, message);
    expect_clean(reader);
    expect(reader.read_file("missing", 0) == -1, "failed-open reader cannot read");
    expect(reader.close() == 0, "close after failed open");
}

static void expect_bad_crc(const Fixture& f, const Entry& e, const char* message)
{
    FixtureFile file;
    if (!file.save(f))
        return;
    pnnx::StoreZipReader reader;
    const int result = reader.open(file.path);
    expect(result == 0, "CRC is checked on consumption, not open");
    if (result == 0)
    {
        std::vector<char> output(e.value.size());
        expect(reader.read_file(e.name, output.empty() ? 0 : output.data()) == -1, message);
    }
}

static void test_valid_archives()
{
    expect(fixture_crc("123456789") == 0xcbf43926u && fixture_crc("") == 0, "CRC oracle known vectors");
    std::vector<Entry> entries;
    entries.push_back(Entry("model/data.pkl", "123456789"));
    entries.push_back(Entry("model/empty", ""));
    entries.push_back(Entry("model/data/0", std::string("\0\x80\xff\n", 4)));
    const std::string zero_crc("\x9d\x0a\xd9\x6d", 4);
    expect(fixture_crc(zero_crc) == 0, "nonempty zero-CRC known vector");
    entries.push_back(Entry("model/zero_crc", zero_crc));
    entries[0].local_extra = extra_field(0x4246, Bytes(12, 'Z')); // Torch alignment padding
    entries[0].central_extra = extra_field(0xcafe, Bytes());
    entries[0].comment = "entry comment";
    // Reader-only CRC test runs before any StoreZipWriter is constructed.
    expect_valid(make_zip(entries), entries, "normal STORE archive");
    expect_valid(make_zip(entries, false, 0, "comment PK\005\006 with a false EOCD signature"), entries, "archive comment and false signature");
    expect_valid(make_zip(entries, false, 0, std::string(65535, 'c')), entries, "maximum EOCD comment");

    const std::vector<Entry> empty;
    expect_valid(make_zip(empty), empty, "empty normal archive");
    expect_valid(make_zip(empty, true), empty, "empty ZIP64 archive");

    for (unsigned int fields = 1; fields < 16; fields++)
    {
        std::vector<Entry> one(1, Entry("conditional", "123456789"));
        one[0].zip64_fields = fields;
        expect_valid(make_zip(one), one, "conditional central ZIP64 fields including disk-only");
    }
    for (unsigned int fields = 1; fields < 4; fields++)
    {
        std::vector<Entry> one(1, Entry("local64", "123456789"));
        one[0].local_zip64_fields = fields;
        expect_valid(make_zip(one), one, "conditional local ZIP64 sizes");
    }

    for (unsigned int field = 1; field <= 32; field <<= 1)
        expect_valid(make_zip(entries, true, field), entries, "any single saturated EOCD field activates ZIP64");
    expect_valid(make_zip(entries, true, 0), entries, "unsaturated EOCD with optional ZIP64 trailer");

    Fixture extensible = make_zip(entries, true);
    extensible.bytes.insert(extensible.bytes.begin() + extensible.locator, 8, 0);
    put(extensible.bytes, extensible.zip64 + 4, 52, 8);
    expect_valid(extensible, entries, "bounded ZIP64 extensible data sector");

    for (unsigned int mode = 0; mode < 8; mode++)
    {
        std::vector<Entry> one(1, Entry("model/data/0", "123456789"));
        one[0].flags = 0x0808; // UTF-8 and data descriptor in BOTH headers
        one[0].descriptor_signature = (mode & 1) != 0;
        one[0].descriptor64 = (mode & 2) != 0;
        one[0].local_zip64_fields = (mode & 4) ? 3 : 0;
        one[0].local_extra = extra_field(0x4246, Bytes(4, 'Z'));
        expect_valid(make_zip(one), one, "Torch-style bit3 placeholders and data descriptor");
    }
}

static void test_crc()
{
    const Entry e("crc", "123456789");
    const Fixture base = make_zip(std::vector<Entry>(1, e));
    Fixture f = base;
    f.bytes[f.data[0]] ^= 1;
    expect_bad_crc(f, e, "corrupt consumed data");
    f = base;
    put(f.bytes, f.local[0] + 14, 1, 4);
    put(f.bytes, f.central[0] + 16, 1, 4);
    expect_bad_crc(f, e, "matching headers with wrong CRC");

    // torch.serialization.set_crc32_options(False) is NOT an integrity bypass.
    // Fixtures meant to be consumed must enable CRC; zero is still verified.
    f = base;
    put(f.bytes, f.local[0] + 14, 0, 4);
    put(f.bytes, f.central[0] + 16, 0, 4);
    expect_bad_crc(f, e, "CRC-disabled nonempty STORE entry is rejected on read");

    Entry descriptor = e;
    descriptor.flags = 8;
    f = make_zip(std::vector<Entry>(1, descriptor));
    put(f.bytes, f.central[0] + 16, 0, 4);
    expect_bad_crc(f, descriptor, "zero central CRC with bit3 is verified");

    const Entry empty("empty", "");
    f = make_zip(std::vector<Entry>(1, empty));
    put(f.bytes, f.local[0] + 14, 1, 4);
    put(f.bytes, f.central[0] + 16, 1, 4);
    expect_bad_crc(f, empty, "empty entry with nonzero CRC fails even with null buffer");
}

static void test_directory_bounds()
{
    const Fixture base = make_zip(std::vector<Entry>(1, Entry("a", "123456789")));
    const size_t lengths[] = {0, 4, 21, base.eocd, base.bytes.size() - 1};
    for (size_t i = 0; i < sizeof(lengths) / sizeof(lengths[0]); i++)
    {
        Fixture f = base;
        f.bytes.resize(lengths[i]);
        expect_rejected(f, "truncated or missing EOCD");
    }
    Fixture f = base;
    put(f.bytes, f.eocd, 0, 4);
    expect_rejected(f, "missing EOCD signature");
    f = base;
    put(f.bytes, f.eocd + 20, 1, 2);
    expect_rejected(f, "EOCD comment exceeds archive");
    f = base;
    f.bytes.push_back(0);
    expect_rejected(f, "unaccounted trailing bytes");

    const size_t count_fields[] = {4, 6, 8, 10, 12, 16};
    for (size_t i = 0; i < 6; i++)
    {
        f = base;
        put(f.bytes, f.eocd + count_fields[i], i < 4 ? 0xffff : 0xffffffffu, i < 4 ? 2 : 4);
        expect_rejected(f, "any saturated EOCD field requires a ZIP64 locator");
    }

    f = base;
    put(f.bytes, f.eocd + 16, f.eocd + 1, 4);
    expect_rejected(f, "CD offset after EOCD");
    f = base;
    put(f.bytes, f.eocd + 12, f.cd_size + 1, 4);
    expect_rejected(f, "CD size overlaps EOCD");
    f = base;
    put(f.bytes, f.eocd + 16, 0xfffffffeu, 4);
    put(f.bytes, f.eocd + 12, 0xfffffffeu, 4);
    expect_rejected(f, "CD addition cannot wrap");
    f = base;
    put(f.bytes, f.eocd + 8, 2, 2);
    put(f.bytes, f.eocd + 10, 2, 2);
    expect_rejected(f, "record count exceeds CD size / 46");
    f = base;
    put(f.bytes, f.eocd + 12, 45, 4);
    expect_rejected(f, "CD too short for a fixed header");
    f = base;
    put(f.bytes, f.eocd + 8, 0, 2);
    put(f.bytes, f.eocd + 10, 0, 2);
    expect_rejected(f, "zero count cannot hide a nonempty directory");

    const Fixture empty = make_zip(std::vector<Entry>());
    f = empty;
    put(f.bytes, f.eocd + 16, 1, 4);
    expect_rejected(f, "zero-count CD offset still bounded by EOCD");
    f = empty;
    put(f.bytes, f.eocd + 12, 1, 4);
    expect_rejected(f, "zero-count CD size still bounded by EOCD");

    std::vector<Entry> two;
    two.push_back(Entry("a", "a"));
    two.push_back(Entry("b", "b"));
    f = make_zip(two);
    put(f.bytes, f.eocd + 8, 1, 2);
    put(f.bytes, f.eocd + 10, 1, 2);
    expect_rejected(f, "count cannot hide trailing central entries");

    for (size_t field = 28; field <= 32; field += 2)
    {
        f = base;
        put(f.bytes, f.central[0] + field, 65535, 2);
        expect_rejected(f, "central name/extra/comment must fit CD size");
    }
    f = base;
    put(f.bytes, f.central[0], 0, 4);
    expect_rejected(f, "bad central signature");
    f = base;
    put(f.bytes, f.local[0], 0, 4);
    expect_rejected(f, "bad local signature");
    f = base;
    put(f.bytes, f.central[0] + 42, f.cd - 29, 4);
    expect_rejected(f, "local fixed header overlaps CD");
    f = base;
    put(f.bytes, f.local[0] + 28, 65535, 2);
    expect_rejected(f, "local variable header overlaps CD");
    f = base;
    put(f.bytes, f.local[0] + 18, 10, 4);
    put(f.bytes, f.local[0] + 22, 10, 4);
    put(f.bytes, f.central[0] + 20, 10, 4);
    put(f.bytes, f.central[0] + 24, 10, 4);
    expect_rejected(f, "entry data overlaps CD by one byte, but fits archive");
}

static void test_zip64_bounds_and_disks()
{
    const uint64_t huge = (std::numeric_limits<uint64_t>::max)();
    const std::vector<Entry> one(1, Entry("a", "123456789"));
    const Fixture base = make_zip(one, true);
    Fixture f = base;
    put(f.bytes, f.zip64 + 24, huge, 8);
    put(f.bytes, f.zip64 + 32, huge, 8);
    expect_rejected(f, "huge ZIP64 record count in a tiny file");
    f = base;
    put(f.bytes, f.zip64 + 40, huge, 8);
    expect_rejected(f, "huge ZIP64 CD size");
    f = base;
    put(f.bytes, f.zip64 + 48, huge, 8);
    expect_rejected(f, "huge ZIP64 CD offset");
    f = base;
    put(f.bytes, f.zip64 + 40, f.cd_size + 1, 8);
    expect_rejected(f, "CD overlaps ZIP64 EOCD, not just legacy EOCD");
    f = make_zip(std::vector<Entry>(), true);
    put(f.bytes, f.zip64 + 48, 1, 8);
    expect_rejected(f, "zero-entry ZIP64 CD offset is still bounded");
    f = make_zip(std::vector<Entry>(), true);
    put(f.bytes, f.zip64 + 40, 1, 8);
    expect_rejected(f, "zero-entry ZIP64 CD size is still bounded");
    f = base;
    put(f.bytes, f.locator + 8, huge, 8);
    expect_rejected(f, "ZIP64 locator offset cannot overflow");
    f = base;
    put(f.bytes, f.locator + 8, f.locator - 55, 8);
    expect_rejected(f, "ZIP64 fixed record overlaps locator");
    f = base;
    put(f.bytes, f.zip64 + 4, 43, 8);
    expect_rejected(f, "ZIP64 record below minimum size");
    f = base;
    put(f.bytes, f.zip64 + 4, huge, 8);
    expect_rejected(f, "ZIP64 extensible record size cannot overflow");
    f = base;
    put(f.bytes, f.zip64 + 4, 45, 8);
    expect_rejected(f, "ZIP64 record overlaps locator");
    f = base;
    put(f.bytes, f.locator, 0, 4);
    expect_rejected(f, "bad locator signature");
    f = base;
    put(f.bytes, f.zip64, 0, 4);
    expect_rejected(f, "bad ZIP64 EOCD signature");

    const size_t disk_fields[] = {base.locator + 4, base.locator + 16, base.zip64 + 16, base.zip64 + 20};
    for (size_t i = 0; i < 4; i++)
    {
        f = base;
        put(f.bytes, disk_fields[i], i == 1 ? 2 : 1, 4);
        expect_rejected(f, "ZIP64 multi-disk archives rejected");
    }
    f = base;
    put(f.bytes, f.locator + 16, 0, 4);
    expect_rejected(f, "zero ZIP64 disk count");
    f = base;
    put(f.bytes, f.zip64 + 24, 0, 8);
    expect_rejected(f, "ZIP64 per-disk count differs from total");

    const Fixture normal = make_zip(one);
    for (size_t field = 4; field <= 6; field += 2)
    {
        f = normal;
        put(f.bytes, f.eocd + field, 1, 2);
        expect_rejected(f, "normal EOCD multi-disk archive rejected");
    }
    f = normal;
    put(f.bytes, f.eocd + 8, 0, 2);
    expect_rejected(f, "normal per-disk count differs from total");
    f = normal;
    put(f.bytes, f.central[0] + 34, 1, 2);
    expect_rejected(f, "entry starts on another disk");
    std::vector<Entry> other_disk = one;
    other_disk[0].zip64_fields = Zip64Disk;
    other_disk[0].disk = 1;
    expect_rejected(make_zip(other_disk), "conditional ZIP64 entry disk must be zero");

    f = make_zip(one, true, 16); // only CD size saturated
    put(f.bytes, f.eocd + 16, f.cd + 1, 4);
    expect_rejected(f, "unsaturated EOCD values must agree with ZIP64");
}

static void test_names_flags_and_local_metadata()
{
    std::vector<Entry> entries(2, Entry("duplicate", "123456789"));
    expect_rejected(make_zip(entries), "duplicate central names");
    entries.pop_back();
    entries[0].name = std::string("a\0b", 3);
    expect_rejected(make_zip(entries), "embedded NUL in matching names");

    const Fixture base = make_zip(std::vector<Entry>(1, Entry("abc", "123456789")));
    Fixture f = base;
    f.bytes[f.local[0] + 30] = 'z';
    expect_rejected(f, "local filename bytes must match central filename");
    f = base;
    f.bytes[f.local[0] + 30] = 0;
    expect_rejected(f, "embedded NUL in local filename");
    f = base;
    put(f.bytes, f.local[0] + 26, 2, 2);
    expect_rejected(f, "local filename length must match central filename");
    f = base;
    put(f.bytes, f.local[0] + 8, 8, 2);
    expect_rejected(f, "local compression method must match central method");
    f = base;
    put(f.bytes, f.local[0] + 6, 8, 2);
    expect_rejected(f, "local-only bit3 is a flag mismatch, not a valid descriptor");
    f = base;
    put(f.bytes, f.central[0] + 8, 0x800, 2);
    expect_rejected(f, "central-only UTF-8 flag mismatch");

    const uint16_t encrypted_flags[] = {1, 0x40, 0x2000};
    for (size_t i = 0; i < 3; i++)
    {
        for (unsigned int headers = 1; headers <= 3; headers++)
        {
            f = base;
            if (headers & 1)
                put(f.bytes, f.local[0] + 6, encrypted_flags[i], 2);
            if (headers & 2)
                put(f.bytes, f.central[0] + 8, encrypted_flags[i], 2);
            expect_rejected(f, "encryption/masked-header flags in either header");
        }
    }
    f = base;
    put(f.bytes, f.central[0] + 24, 8, 4);
    expect_rejected(f, "STORE compressed/uncompressed size mismatch");
    f = base;
    put(f.bytes, f.local[0] + 22, 8, 4);
    expect_rejected(f, "non-bit3 local size mismatch");
    f = base;
    put(f.bytes, f.local[0] + 14, 0, 4);
    expect_rejected(f, "non-bit3 local CRC mismatch");

    Entry descriptor("descriptor", "123456789");
    descriptor.flags = 8;
    const std::vector<Entry> descriptor_entries(1, descriptor);
    const Fixture descriptor_base = make_zip(descriptor_entries);
    const size_t descriptor_fields[] = {14, 18, 22};
    for (size_t i = 0; i < 3; i++)
    {
        f = descriptor_base;
        put(f.bytes, f.local[0] + descriptor_fields[i], 1, 4);
        expect_rejected(f, "bit3 does not excuse a conflicting nonzero local CRC or size");
    }
    f = descriptor_base;
    put(f.bytes, f.local[0] + 14, fixture_crc(descriptor.value), 4);
    put(f.bytes, f.local[0] + 18, descriptor.payload.size(), 4);
    put(f.bytes, f.local[0] + 22, descriptor.value.size(), 4);
    expect_valid(f, descriptor_entries, "bit3 also allows matching nonzero local metadata");

    descriptor.local_zip64_fields = Zip64Uncompressed | Zip64Compressed;
    const Fixture descriptor64 = make_zip(std::vector<Entry>(1, descriptor));
    for (size_t field = 4; field <= 12; field += 8)
    {
        f = descriptor64;
        put(f.bytes, f.local_extra[0] + field, 1, 8);
        expect_rejected(f, "bit3 ZIP64 local sizes must be placeholders or match central metadata");
    }
}

static void test_extra_fields()
{
    const Entry original("extra", "123456789");
    for (unsigned int local = 0; local < 2; local++)
    {
        for (size_t trailing = 1; trailing < 4; trailing++)
        {
            Entry e = original;
            (local ? e.local_extra : e.central_extra) = Bytes(trailing, 0);
            expect_rejected(make_zip(std::vector<Entry>(1, e)), "extra trailing bytes cannot form a TLV header");
        }
        Entry e = original;
        Bytes malformed = extra_field(0xcafe, Bytes(1, 0));
        put(malformed, 2, 2, 2);
        (local ? e.local_extra : e.central_extra) = malformed;
        expect_rejected(make_zip(std::vector<Entry>(1, e)), "unknown extra length exceeds containing field");

        e = original;
        (local ? e.local_zip64_fields : e.zip64_fields) = Zip64Uncompressed;
        (local ? e.local_extra : e.central_extra) = extra_field(0xcafe, Bytes(8, 0));
        Fixture f = make_zip(std::vector<Entry>(1, e));
        put(f.bytes, (local ? f.local_extra[0] : f.central_extra[0]) + 2, 4, 2);
        expect_rejected(f, "ZIP64 value cannot borrow bytes beyond its own subfield");

        e = original;
        (local ? e.local_zip64_fields : e.zip64_fields) = Zip64Uncompressed;
        (local ? e.local_extra : e.central_extra) = Bytes(1, 0);
        expect_rejected(make_zip(std::vector<Entry>(1, e)), "validate malformed trailing extras after ZIP64");
        (local ? e.local_extra : e.central_extra) = extra_field(1, Bytes());
        expect_rejected(make_zip(std::vector<Entry>(1, e)), "duplicate ZIP64 subfields are ambiguous");

        e = original;
        (local ? e.local_extra : e.central_extra) = extra_field(0xcafe, Bytes(65531, 'Z'));
        const std::vector<Entry> one(1, e);
        expect_valid(make_zip(one), one, "maximum bounded harmless extra field");
    }

    const Fixture base = make_zip(std::vector<Entry>(1, original));
    const size_t central_fields[] = {20, 24, 42, 34};
    for (size_t i = 0; i < 4; i++)
    {
        Fixture f = base;
        put(f.bytes, f.central[0] + central_fields[i], i == 3 ? 0xffff : 0xffffffffu, i == 3 ? 2 : 4);
        expect_rejected(f, "required conditional central ZIP64 field is missing");
    }
    for (size_t field = 18; field <= 22; field += 4)
    {
        Fixture f = base;
        put(f.bytes, f.local[0] + field, 0xffffffffu, 4);
        expect_rejected(f, "required local ZIP64 size is missing");
    }
    Entry disk = original;
    disk.zip64_fields = Zip64Disk;
    disk.central_extra = extra_field(0xcafe, Bytes(8, 0));
    Fixture f = make_zip(std::vector<Entry>(1, disk));
    put(f.bytes, f.central_extra[0] + 2, 3, 2);
    expect_rejected(f, "conditional disk needs four bytes within ZIP64 subfield");

    Entry sizes = original;
    sizes.zip64_fields = Zip64Uncompressed | Zip64Compressed;
    f = make_zip(std::vector<Entry>(1, sizes));
    put(f.bytes, f.central_extra[0] + 4, (std::numeric_limits<uint64_t>::max)(), 8);
    put(f.bytes, f.central_extra[0] + 12, (std::numeric_limits<uint64_t>::max)(), 8);
    expect_rejected(f, "huge ZIP64 STORE payload cannot wrap into the directory");
    sizes.zip64_fields = Zip64Offset;
    f = make_zip(std::vector<Entry>(1, sizes));
    put(f.bytes, f.central_extra[0] + 4, (std::numeric_limits<uint64_t>::max)(), 8);
    expect_rejected(f, "huge ZIP64 local offset cannot wrap into archive");
}

static void test_unused_compression()
{
    std::vector<Entry> entries;
    entries.push_back(Entry("model.json", "{}"));
    entries.push_back(Entry("unused.deflate", "123456789"));
    entries[1].method = 8;
    // A literal raw-DEFLATE stored block, not a compressor implementation.
    entries[1].payload = std::string("\x01\x09\x00\xf6\xff", 5) + "123456789";
    FixtureFile file;
    if (!file.save(make_zip(entries)))
        return;
    pnnx::StoreZipReader reader;
    expect(reader.open(file.path) == 0, "unused compressed attachment is accepted at open");
    char output[2] = {};
    expect(reader.read_file(entries[0].name, output) == 0 && std::string(output, 2) == "{}", "STORE sibling of compressed attachment is readable");
    char untouched[9];
    memset(untouched, 'x', sizeof(untouched));
    expect(reader.read_file(entries[1].name, untouched) == -1 && std::string(untouched, sizeof(untouched)) == std::string(9, 'x'), "unsupported consumed compression fails before accessing output");
    expect(reader.close() == 0, "close mixed archive");

    entries[1].method = 99;
    entries[1].flags = 8;
    entries[1].zip64_fields = Zip64Uncompressed;
    Fixture huge = make_zip(entries);
    put(huge.bytes, huge.central_extra[1] + 4, (std::numeric_limits<uint64_t>::max)(), 8);
    if (!file.save(huge))
        return;
    expect(reader.open(file.path) == 0, "unused method with huge advertised uncompressed size does not allocate payload");
    expect(reader.read_file(entries[1].name, 0) == -1, "unknown method refused without allocating a destination");
}

static void test_lifecycle_and_allocation_failures()
{
    std::vector<Entry> entries;
    entries.push_back(Entry(std::string(80, 'a'), "first"));
    entries.push_back(Entry(std::string(80, 'b'), "second"));
    entries[0].local_extra = extra_field(0x4246, Bytes(8, 'Z'));
    const Fixture good = make_zip(entries);
    FixtureFile file;
    FixtureFile bad_file("test_storezip_bad_fixture.zip");
    Fixture bad = good;
    bad.bytes[bad.local[1] + 30] = 'c'; // fail after the first map insertion
    if (!file.save(good) || !bad_file.save(bad))
        return;

    pnnx::StoreZipReader reader;
    expect(reader.open(file.path) == 0, "open before lifecycle checks");
    expect(reader.read_file(entries[0].name, 0) == -1, "null nonempty destination fails safely");
    expect(reader.open(bad_file.path) == -1, "failed reopen clears previous and partially read metadata");
    expect_clean(reader);
    expect(reader.open(file.path) == 0, "reader recovers after malformed archive");
    expect(reader.close() == 0, "reader close");
    expect_clean(reader);
    expect(reader.read_file(entries[0].name, 0) == -1, "no stale metadata after close");
    expect(reader.open(file.path) == 0, "open before missing-file check");
    expect(remove(bad_file.path.c_str()) == 0, "remove owned fixture for missing-file check");
    expect(reader.open(bad_file.path) == -1, "fopen failure also leaves reader clean");
    expect_clean(reader);

    bool completed = false;
    int injected_failures = 0;
    for (int budget = 0; budget < 64; budget++)
    {
        allocation_budget = budget;
        int result = -2;
        bool escaped = false;
        try
        {
            result = reader.open(file.path);
        }
        catch (...)
        {
            escaped = true;
        }
        allocation_budget = -1;
        expect(!escaped, "metadata allocation exception must not escape open");
        if (result == 0)
        {
            completed = true;
            expect_contents(reader, entries);
            reader.close();
            break;
        }
        expect(result == -1, "allocation failure returns error");
        expect_clean(reader);
        injected_failures++;
    }
    expect(completed && injected_failures >= 5, "allocation failures cover tail, names, extras, and partially populated map");

    bad = make_zip(entries, true);
    put(bad.bytes, bad.zip64 + 24, (std::numeric_limits<uint64_t>::max)(), 8);
    put(bad.bytes, bad.zip64 + 32, (std::numeric_limits<uint64_t>::max)(), 8);
    if (!bad_file.save(bad))
        return;
    allocation_attempts = 0;
    largest_allocation = 0;
    const int result = reader.open(bad_file.path);
    const size_t attempts = allocation_attempts;
    const size_t largest = largest_allocation;
    // MSVC debug containers may also allocate a small iterator proxy for tail.
    expect(result == -1 && attempts <= 4 && largest <= 65557, "malicious count rejected after only bounded tail allocations");
    expect_clean(reader);

    // Deterministic short-read path without truncating an open OS handle.
    if (reader.open(file.path) == 0)
    {
        reader.filemetas[entries[0].name].offset = good.bytes.size() - 1;
        reader.filemetas[entries[0].name].size = 4;
        char buffer[4];
        expect(reader.read_file(entries[0].name, buffer) == -1, "short read is propagated");
        reader.close();
    }
    else
    {
        expect(false, "open for short-read check");
    }
}

static void test_writer()
{
    static_assert(sizeof(pnnx::StoreZipWriter::StoreZipMeta::lfh_offset) == 8, "writer metadata uses 64-bit offsets");
    FixtureFile first("test_storezip_writer_first.zip");
    FixtureFile second("test_storezip_writer_second.zip");
    pnnx::StoreZipWriter writer;
    expect(writer.write_file("closed", 0, 0) == -1, "write before open fails");
    expect(writer.open(first.path) == 0, "open writer");
    expect(writer.write_file("a", "123456789", 9) == 0, "write nonempty STORE");
    expect(writer.write_file("empty", 0, 0) == 0, "write empty STORE with null buffer");
    expect(writer.write_file("a", "x", 1) == -1, "writer refuses duplicate names");
    expect(writer.write_file(std::string("a\0b", 3), "x", 1) == -1, "writer refuses embedded NUL");
    expect(writer.write_file(std::string(65536, 'n'), 0, 0) == -1, "writer refuses overflowing filename length");
    expect(writer.write_file("null", 0, 1) == -1, "writer refuses null nonempty data");
    expect(writer.write_file("overflow", "x", (std::numeric_limits<uint64_t>::max)()) == -1, "writer rejects unrepresentable payload before CRC or I/O");
    expect(writer.write_file("overflow", "x", (uint64_t)(std::numeric_limits<int64_t>::max)()) == -1, "writer accounts for local header when bounding final payload offset");
    expect(writer.filemetas.size() == 2, "rejected writes leave metadata unchanged");
    expect(writer.open(second.path) == 0 && writer.filemetas.empty(), "writer reopen closes old archive and clears metadata");
    expect(writer.write_file("b", "next", 4) == 0, "write after reopen");
    expect(writer.close() == 0 && writer.close() == 0, "idempotent writer close");
    expect(writer.fp == 0 && writer.filemetas.empty(), "writer close clears state");

    pnnx::StoreZipReader reader;
    expect(reader.open(first.path) == 0, "read writer's ZIP64 format, including surplus local extra bytes");
    std::vector<Entry> expected;
    expected.push_back(Entry("a", "123456789"));
    expected.push_back(Entry("empty", ""));
    expect_contents(reader, expected);
    expect(reader.open(second.path) == 0, "read reopened writer's archive");
    expect_contents(reader, std::vector<Entry>(1, Entry("b", "next")));
    reader.close();

    expect(writer.open(second.path) == 0 && writer.close() == 0, "write empty ZIP64 archive");
    expect(reader.open(second.path) == 0 && reader.get_names().empty(), "read empty self-written ZIP64 archive");
    reader.close();

    expect(writer.open(second.path) == 0, "open before failed-writer state test");
    writer.failed = true; // exercise sticky I/O failure without unsafe OS devices
    expect(writer.write_file("x", "x", 1) == -1, "sticky write failure propagates");
    expect(writer.close() == -1, "sticky failure propagates through close");
    expect(writer.fp == 0 && writer.filemetas.empty(), "failed writer close cleans up");

    expect(writer.open(second.path) == 0, "writer recovers after failure");
    const std::string long_name(80, 'm');
    allocation_budget = 0;
    const int result = writer.write_file(long_name, "x", 1);
    allocation_budget = -1;
    expect(result == -1 && writer.filemetas.empty(), "writer metadata allocation failure is recoverable");
    expect(writer.write_file(long_name, "x", 1) == 0 && writer.close() == 0, "retry write after allocation failure");
    expect(reader.open(second.path) == 0, "writer allocation failure leaves no partial local header");
    expect_contents(reader, std::vector<Entry>(1, Entry(long_name, "x")));
    reader.close();

    // Exercise every allocation site, including vector growth and the name
    // copy inside push_back, using a fresh writer to avoid retained capacity.
    bool completed = false;
    int injected_failures = 0;
    for (int budget = 0; budget < 32; budget++)
    {
        pnnx::StoreZipWriter fresh;
        expect(fresh.open(second.path) == 0, "open writer for allocation sweep");
        allocation_budget = budget;
        int write_result = -2;
        bool escaped = false;
        try
        {
            write_result = fresh.write_file(long_name, "x", 1);
        }
        catch (...)
        {
            escaped = true;
        }
        allocation_budget = -1;
        expect(!escaped, "writer allocation exception must not escape write_file");
        if (write_result != 0)
        {
            injected_failures++;
            expect(write_result == -1 && fresh.filemetas.empty() && !fresh.failed, "writer allocation failure preserves retryable state");
            expect(fresh.write_file(long_name, "x", 1) == 0, "retry each failed writer allocation site");
        }
        expect(fresh.close() == 0 && fresh.fp == 0 && fresh.filemetas.empty(), "close writer after allocation sweep");
        expect(reader.open(second.path) == 0, "allocation sweep leaves a valid archive");
        expect_contents(reader, std::vector<Entry>(1, Entry(long_name, "x")));
        reader.close();
        if (write_result == 0)
        {
            completed = true;
            break;
        }
    }
    expect(completed && injected_failures >= 3, "writer sweep covers metadata name, vector allocation, and stored name copy");
}

int main()
{
    test_valid_archives();
    test_crc();
    test_directory_bounds();
    test_zip64_bounds_and_disks();
    test_names_flags_and_local_metadata();
    test_extra_fields();
    test_unused_compression();
    test_lifecycle_and_allocation_failures();
    test_writer();
    expect(oversized_allocation_attempts == 0, "no test requested a large allocation");
    if (test_failures)
        fprintf(stderr, "%d storezip test(s) failed\n", test_failures);
    return test_failures ? 1 : 0;
}