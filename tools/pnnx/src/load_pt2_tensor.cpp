// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// tensor_meta materialization (split out of load_exportedprogram.cpp).

#include "load_pt2_tensor.h"

#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "ir.h"
#include "pnnx_json.h"
#include "storezip.h"

namespace pnnx {

// materialize the logical row-major tensor described by the serialized
// tensor_meta (sizes / strides / storage_offset) out of a raw storage buffer
// into an Attribute, so transposed, sliced, or shared-storage views are stored
// correctly. shared by the 2.8+ archive path (raw byte records) and the
// legacy(<2.8) path (raw storage shards from a pickled state dict). raw is
// taken by value because the contiguous path may resize it.
int load_tensor_from_raw(std::vector<char> raw, const JsonValue& meta, Attribute& a)
{
    // parse serialized sizes / strides / storage_offset from tensor_meta
    std::vector<int> sizes;
    std::vector<int64_t> strides;
    int64_t storage_offset = 0;
    if (meta.is_object())
    {
        if (meta.has("sizes"))
        {
            const JsonValue& s = meta["sizes"];
            for (size_t i = 0; i < s.size(); i++)
            {
                if (s[i].has("as_int"))
                    sizes.push_back((int)s[i]["as_int"].as_int());
                else
                    sizes.push_back(-1);
            }
        }
        if (meta.has("strides"))
        {
            const JsonValue& st = meta["strides"];
            for (size_t i = 0; i < st.size(); i++)
            {
                if (st[i].has("as_int"))
                    strides.push_back(st[i]["as_int"].as_int());
                else
                    strides.push_back(0);
            }
        }
        if (meta.has("storage_offset") && meta["storage_offset"].has("as_int"))
            storage_offset = meta["storage_offset"]["as_int"].as_int();
    }

    const int dims = (int)sizes.size();
    if (dims == 0 || strides.size() != sizes.size())
    {
        if (dims == 0 && strides.empty())
        {
            // rank-zero scalar that is a view into a shared storage (e.g.
            // base[3]): empty sizes/strides with a nonzero storage_offset are
            // valid metadata selecting one element. keep just that element so
            // the scalar attribute reads the selected storage slot instead of
            // the whole backing storage (which would read element 0).
            const int es = (int)a.elemsize();
            if (es > 0 && storage_offset >= 0 && raw.size() >= (size_t)es)
            {
                const uint64_t off = (uint64_t)storage_offset * (uint64_t)es;
                if (off <= raw.size() - (size_t)es)
                {
                    a.data.assign(raw.begin() + (size_t)off, raw.begin() + (size_t)off + (size_t)es);
                    return 0;
                }
            }
        }
        // no usable tensor_meta: the declared shape cannot be satisfied
        // consistently, so reject instead of attaching mismatched bytes
        a.data.clear();
        return -1;
    }

    bool symbolic = false;
    size_t count = 1;
    for (int i = 0; i < dims; i++)
    {
        // a zero-sized dim is a valid *empty* tensor (e.g. an empty view of a
        // shared nonempty storage): its logical element count is zero, not
        // symbolic. only negative sizes (the loader's dynamic marker) are
        // unresolvable and fall back to the raw storage.
        if (sizes[i] == 0)
        {
            count = 0;
            break;
        }
        if (sizes[i] < 0)
        {
            symbolic = true;
            break;
        }
        if (count > (size_t)-1 / (size_t)sizes[i])
        {
            symbolic = true; // product overflow
            break;
        }
        count *= (size_t)sizes[i];
    }

    if (count == 0)
    {
        // empty tensor: an empty view of shared storage must not drag the
        // whole backing buffer into the attribute (it would bloat the archive
        // and duplicate the storage); the caller has already recorded the
        // declared shape, so materialize a zero-byte attribute
        a.data.clear();
        return 0;
    }

    if (symbolic)
    {
        // a dynamic/unresolvable dimension cannot be materialized as a static
        // constant; reject the archive instead of installing mismatched bytes
        a.data.clear();
        return -1;
    }

    const int elemsize = (int)a.elemsize();
    if (elemsize <= 0)
    {
        // unknown/unsupported dtype maps to type 0 with no element size
        a.data.clear();
        return -1;
    }

    // the logical tensor is a view of this storage: materializing must never
    // grow the element count beyond what the raw storage holds (only expand/
    // stride tricks can do that, and those carry no data here). an exaggerated
    // tensor_meta (corrupt / hostile sizes) would otherwise make resize/
    // vector-alloc below explode into gigabytes and OOM the process.
    const size_t raw_elems_total = raw.size() / (size_t)elemsize; // full elems in raw
    if (count > raw_elems_total)
    {
        // a zero-stride (expanded) view legitimately repeats elements, so its
        // logical count may exceed the backing storage; materialization below
        // repeats through stride 0 and bounds-checks every source offset.
        bool has_zero_stride = false;
        for (int i = 0; i < dims; i++)
        {
            if (strides[i] == 0)
            {
                has_zero_stride = true;
                break;
            }
        }
        if (has_zero_stride)
        {
            // a genuine expanded view (torch.tensor([1.]).expand(n), a
            // broadcast buffer) has exactly one source element per output run;
            // verify count == the zero-stride expansion product so an
            // inconsistent meta cannot pass. a hostile meta may still ask for a
            // huge count, but that is bounded by the absolute byte cap below
            // (shared with the materialization path).
            if (raw_elems_total == 0)
            {
                a.data.clear();
                return -1;
            }
            size_t expanded = raw_elems_total;
            for (int i = 0; i < dims; i++)
            {
                if (strides[i] == 0 && sizes[i] > 1)
                {
                    if (expanded > (size_t)-1 / (size_t)sizes[i])
                    {
                        a.data.clear();
                        return -1;
                    }
                    expanded *= (size_t)sizes[i];
                }
            }
            if (count != expanded)
            {
                // inconsistent with any zero-stride expansion
                a.data.clear();
                return -1;
            }
        }
        else
        {
            // overlapping view with only non-zero strides: the logical element
            // count may exceed the backing storage while every element address
            // is still valid (e.g. base.as_strided((3,3),(1,1)) over a
            // five-element storage, or a negative-stride flip view). materialize
            // below repeats the overlapping reads, so only reject when an extreme
            // reachable address escapes the storage - computed here in O(dims),
            // before any allocation (a hostile meta can neither OOM us nor pass
            // an out-of-range address).
            int64_t min_addr = storage_offset;
            int64_t max_addr = storage_offset;
            for (int i = 0; i < dims && min_addr >= 0 && max_addr < (int64_t)raw_elems_total; i++)
            {
                const int64_t extent = (int64_t)sizes[i] - 1;
                const int64_t st = (int64_t)strides[i];
                if (extent <= 0 || st == 0)
                    continue;
                // safe absolute value (st == INT64_MIN would overflow -st)
                const uint64_t as = st < 0 ? (uint64_t)(-(st + 1)) + 1 : (uint64_t)st;
                // if this dimension alone reaches >= storage the view is OOB;
                // compare via ceil(raw/as) so extent*as cannot overflow
                const uint64_t need = ((uint64_t)raw_elems_total + as - 1) / as;
                if ((uint64_t)extent >= need)
                {
                    a.data.clear();
                    return -1;
                }
                const int64_t span = extent * st;
                if (st >= 0)
                    max_addr += span;
                else
                    min_addr += span;
            }
            if (min_addr < 0 || (uint64_t)max_addr >= (uint64_t)raw_elems_total)
            {
                // some element address would fall outside the storage: the
                // shape/strides are inconsistent with it
                a.data.clear();
                return -1;
            }
            // in-bounds overlap is legitimate (e.g. base.as_strided((3,3),(1,1))
            // over a five-element storage), but the materialization loop below
            // is O(count): a pathological box wholly inside a small storage
            // (e.g. a stride-(1,1) square) makes count quadratic in the storage
            // size and the out-buffer allocation would OOM before any OOB check
            // fires. only materialize a modest multiple of the storage; real
            // overlapping views repeat a handful of elements, never a large
            // fraction of a quadratic blowup.
            if (raw_elems_total == 0 || count / raw_elems_total > 16)
            {
                a.data.clear();
                return -1;
            }
        }
    }

    // already row-major contiguous with zero offset? keep raw
    bool contiguous = storage_offset == 0;
    if (contiguous)
    {
        int64_t expected = 1;
        for (int i = dims - 1; i >= 0; i--)
        {
            if (strides[i] != expected)
            {
                contiguous = false;
                break;
            }
            expected *= sizes[i];
        }
    }
    if (contiguous)
    {
        // clamp the raw storage to the logical tensor size: a contiguous view
        // may share a larger storage, and downstream accessors assume
        // data.size() == elemcount * elemsize
        const size_t expect = count * (size_t)elemsize;
        if (raw.size() > expect)
            raw.resize(expect);
        // raw < expect means the meta claims more bytes than the storage
        // holds; do not fabricate the missing tail - reject the archive
        if (raw.size() != expect)
        {
            a.data.clear();
            return -1;
        }
        a.data = raw;
        return 0;
    }

    // guard against integer overflow when allocating the materialized buffer,
    // and cap the absolute output size (a hostile meta must not drive a
    // multi-gigabyte allocation / OOM)
    if (count > (size_t)-1 / (size_t)elemsize || count * (size_t)elemsize > (size_t)0x40000000)
    {
        a.data.clear();
        return -1;
    }

    // materialize row-major from (sizes, strides, storage_offset), bounds-checking
    // every source offset against the raw storage (values come from the archive).
    // use a division compare so the offset multiply cannot overflow.
    const size_t raw_elems = raw.size() / (size_t)elemsize; // floor: full elems in raw
    std::vector<char> out(count * (size_t)elemsize);
    char* dst = out.data();
    const char* src = raw.data();
    for (size_t n = 0; n < count; n++)
    {
        size_t tmp = n;
        int64_t sto = storage_offset;
        for (int i = dims - 1; i >= 0; i--)
        {
            const int idx = (int)(tmp % (size_t)sizes[i]);
            tmp /= (size_t)sizes[i];
            sto += (int64_t)idx * strides[i];
        }
        if (sto < 0 || (uint64_t)sto >= (uint64_t)raw_elems)
        {
            // out-of-bounds source: this tensor_meta cannot address the raw
            // storage, so the shape/strides are inconsistent with it
            a.data.clear();
            return -1;
        }
        memcpy(dst + n * (size_t)elemsize, src + sto * (size_t)elemsize, (size_t)elemsize);
    }
    a.data = out;
    return 0;
}

// read one weight/constant record (raw storage bytes) from the zip into an
// Attribute (materialization is shared via load_tensor_from_raw). returns 0 on
// success and -1 when the referenced payload record is missing, so the caller
// can reject an incomplete archive instead of installing an empty attribute.
int load_tensor_data(StoreZipReader& zip, const std::vector<std::string>& names,
                     const std::string& dir, const std::string& path_name,
                     const JsonValue& meta, Attribute& a)
{
    std::string record;
    for (size_t j = 0; j < names.size(); j++)
    {
        if (names[j].find(dir + "/" + path_name) != std::string::npos || names[j] == path_name)
        {
            record = names[j];
            break;
        }
    }

    if (record.empty())
    {
        fprintf(stderr, "tensor record %s/%s not found\n", dir.c_str(), path_name.c_str());
        return -1;
    }

    uint64_t size = zip.get_file_size(record);
    std::vector<char> raw((size_t)size);
    if (zip.read_file(record, raw.data()) != 0)
        return -1;

    if (load_tensor_from_raw(raw, meta, a) != 0)
    {
        fprintf(stderr, "tensor record %s cannot be materialized (inconsistent sizes/strides/storage)\n", record.c_str());
        return -1;
    }
    return 0;
}

} // namespace pnnx
