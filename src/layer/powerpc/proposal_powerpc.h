// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PROPOSAL_POWERPC_H
#define LAYER_PROPOSAL_POWERPC_H

#include "proposal.h"

namespace ncnn {

class Proposal_powerpc : public Proposal
{
public:
    Proposal_powerpc();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_PROPOSAL_POWERPC_H
