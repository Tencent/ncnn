// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_UTILS_H
#define PNNX_UTILS_H

namespace pnnx {

unsigned short float32_to_float16(float value);

float float16_to_float32(unsigned short value);

} // namespace pnnx

#endif // PNNX_UTILS_H
