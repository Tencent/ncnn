// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef NCNN_MODELBIN_H
#define NCNN_MODELBIN_H

#include "mat.h"

namespace ncnn {

class DataReader;
class NCNN_EXPORT ModelBin
{
public:
    ModelBin();
    virtual ~ModelBin();
    // element type
    // 0 = auto
    // 1 = float32
    // 2 = float16
    // 3 = int8
    // load vec
    virtual Mat load(int w, int type) const = 0;
    // load image
    virtual Mat load(int w, int h, int type) const;
    // load dim
    virtual Mat load(int w, int h, int c, int type) const;
};

class ModelBinFromDataReaderPrivate;
class NCNN_EXPORT ModelBinFromDataReader : public ModelBin
{
public:
    explicit ModelBinFromDataReader(const DataReader& dr);
    virtual ~ModelBinFromDataReader();

    virtual Mat load(int w, int type) const;

private:
    ModelBinFromDataReader(const ModelBinFromDataReader&);
    ModelBinFromDataReader& operator=(const ModelBinFromDataReader&);

private:
    ModelBinFromDataReaderPrivate* const d;
};

class ModelBinFromMatArrayPrivate;
class NCNN_EXPORT ModelBinFromMatArray : public ModelBin
{
public:
    // construct from weight blob array
    explicit ModelBinFromMatArray(const Mat* weights);
    virtual ~ModelBinFromMatArray();

    virtual Mat load(int w, int type) const;

private:
    ModelBinFromMatArray(const ModelBinFromMatArray&);
    ModelBinFromMatArray& operator=(const ModelBinFromMatArray&);

private:
    ModelBinFromMatArrayPrivate* const d;
};

} // namespace ncnn

#endif // NCNN_MODELBIN_H
