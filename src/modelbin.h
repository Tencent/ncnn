// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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
    virtual Mat load(int w, int type) const;
    // load image
    virtual Mat load(int w, int h, int type) const;
    // load dim
    virtual Mat load(int w, int h, int c, int type) const;
    // load cube
    virtual Mat load(int w, int h, int d, int c, int type) const;
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
