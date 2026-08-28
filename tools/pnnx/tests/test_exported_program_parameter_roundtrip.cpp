// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "ir.h"

#include <stdio.h>

#include <string>

int main()
{
    pnnx::Parameter original;
    original.type = 6;

    const std::string encoded = pnnx::Parameter::encode_to_string(original);
    if (encoded != "[]f")
    {
        fprintf(stderr, "empty float list encoded as %s\n", encoded.c_str());
        return 1;
    }

    const pnnx::Parameter parsed = pnnx::Parameter::parse_from_string(encoded);
    if (parsed.type != 6 || !parsed.af.empty())
    {
        fprintf(stderr, "empty float list decoded with type %d and size %lu\n", parsed.type, (unsigned long)parsed.af.size());
        return 1;
    }

    const std::string roundtrip = pnnx::Parameter::encode_to_string(parsed);
    if (roundtrip != encoded)
    {
        fprintf(stderr, "empty float list roundtrip changed %s to %s\n", encoded.c_str(), roundtrip.c_str());
        return 1;
    }

    return 0;
}
