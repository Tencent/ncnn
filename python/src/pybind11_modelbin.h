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

#ifndef PYBIND11_NCNN_MODELBIN_H
#define PYBIND11_NCNN_MODELBIN_H

#include <modelbin.h>

template<class Base = ncnn::ModelBin>
class PyModelBin : public Base
{
public:
    using Base::Base; // Inherit constructors
    ncnn::Mat load(int w, int type) const override
    {
        PYBIND11_OVERRIDE(ncnn::Mat, Base, load, w, type);
    }
    //ncnn::Mat load(int w, int h, int type) const override {
    //	PYBIND11_OVERRIDE(ncnn::Mat, Base, load, w, h, type);
    //}
    //ncnn::Mat load(int w, int h, int c, int type) const override {
    //	PYBIND11_OVERRIDE(ncnn::Mat, Base, load, w, h, c, type);
    //}
};

template<class Other>
class PyModelBinOther : public PyModelBin<Other>
{
public:
    using PyModelBin<Other>::PyModelBin;
    ncnn::Mat load(int w, int type) const override
    {
        PYBIND11_OVERRIDE(ncnn::Mat, Other, load, w, type);
    }
};

#endif
