/* Tencent is pleased to support the open source community by making ncnn available.
 *
 * Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */

#ifndef PYBIND11_NCNN_BIND_H
#define PYBIND11_NCNN_BIND_H

#include <pybind11/functional.h>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// virtual function pass by reference by https://github.com/pybind/pybind11/issues/2033
#define PYBIND11_OVERRIDE_REFERENCE_IMPL(ret_type, cname, name, ...)                                 \
    do                                                                                               \
    {                                                                                                \
        pybind11::gil_scoped_acquire gil;                                                            \
        pybind11::function override = pybind11::get_override(static_cast<const cname*>(this), name); \
        if (override)                                                                                \
        {                                                                                            \
            auto o = override.operator()<pybind11::return_value_policy::reference>(__VA_ARGS__);     \
            if (pybind11::detail::cast_is_temporary_value_reference<ret_type>::value)                \
            {                                                                                        \
                static pybind11::detail::override_caster_t<ret_type> caster;                         \
                return pybind11::detail::cast_ref<ret_type>(std::move(o), caster);                   \
            }                                                                                        \
            else                                                                                     \
                return pybind11::detail::cast_safe<ret_type>(std::move(o));                          \
        }                                                                                            \
    } while (false)

#define PYBIND11_OVERRIDE_REFERENCE_NAME(ret_type, cname, name, fn, ...)                                    \
    do                                                                                                      \
    {                                                                                                       \
        PYBIND11_OVERRIDE_REFERENCE_IMPL(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, __VA_ARGS__); \
        return cname::fn(__VA_ARGS__);                                                                      \
    } while (false)

#define PYBIND11_OVERRIDE_REFERENCE(ret_type, cname, fn, ...) \
    PYBIND11_OVERRIDE_REFERENCE_NAME(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), #fn, fn, __VA_ARGS__)
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif