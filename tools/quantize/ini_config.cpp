// tpoisonooo is pleased to support the open source community by making ncnn available.
//
// author:tpoisonooo (https://github.com/tpoisonooo/) .
//
// Copyright (C) 2022 tpoisonooo. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "ini_config.h"

namespace ini {
template<>
std::string value_set<std::string>(std::string data)
{
    return "\"" + data + "\"";
}

template<>
std::string value_set<const char*>(const char* data)
{
    return "\"" + std::string(data) + "\"";
}

template<>
std::string value_get<std::string>(std::string text)
{
    auto start = text.find('\"');
    auto end = text.find_last_of('\"');

    return text.substr(start + 1, end - start - 1);
}

} // namespace ini
