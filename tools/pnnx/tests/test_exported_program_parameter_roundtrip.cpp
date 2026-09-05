// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "ir.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>

#include <cmath>
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

    int failures = 0;
    const pnnx::Parameter zero = pnnx::Parameter::parse_from_string(pnnx::Parameter::encode_to_string(pnnx::Parameter(-0.f)));
    if (zero.type != 3 || zero.f != 0.f || !std::signbit(zero.f))
    {
        fprintf(stderr, "negative zero scalar lost its sign\n");
        failures++;
    }

    const pnnx::Parameter zeros = pnnx::Parameter::parse_from_string(pnnx::Parameter::encode_to_string(pnnx::Parameter{-0.f, 0.f}));
    if (zeros.type != 6 || zeros.af.size() != 2 || zeros.af[0] != 0.f || !std::signbit(zeros.af[0]) || zeros.af[1] != 0.f || std::signbit(zeros.af[1]))
    {
        fprintf(stderr, "float list zeros lost their signs\n");
        failures++;
    }

    const double zero64 = strtod(pnnx::double_to_string(-0.0).c_str(), 0);
    if (zero64 != 0.0 || !std::signbit(zero64))
    {
        fprintf(stderr, "negative zero double lost its sign\n");
        failures++;
    }

    return failures ? 1 : 0;
}
