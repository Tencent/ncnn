// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PT2_WEIGHTS_H
#define PNNX_PT2_WEIGHTS_H

#include <map>
#include <string>

#include "ir.h"
#include "pt2_program.h"

namespace pnnx {

class Pt2ArchiveReader;

struct Pt2Weight
{
    Pt2InputSpec::Kind kind;
    Attribute attribute;
};

struct Pt2Weights
{
    std::map<std::string, Pt2Weight> values;
    std::string error;
};

int load_pt2_weights(Pt2ArchiveReader& archive, const Pt2Program& program, Pt2Weights& weights);

} // namespace pnnx

#endif // PNNX_PT2_WEIGHTS_H
