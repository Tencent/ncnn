// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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
