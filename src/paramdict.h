// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_PARAMDICT_H
#define NCNN_PARAMDICT_H

#include "mat.h"

// at most 32 parameters
#define NCNN_MAX_PARAM_COUNT 32

namespace ncnn {

class DataReader;
class Net;
class ParamDictPrivate;
class NCNN_EXPORT ParamDict
{
public:
    // empty
    ParamDict();

    virtual ~ParamDict();

    // copy
    ParamDict(const ParamDict&);

    // assign
    ParamDict& operator=(const ParamDict&);

    // get type, or 0 for an invalid id
    int type(int id) const;

    // getters return the default for an invalid id or mismatched type
    // binary scalar and array types remain untyped int/float data
    // get int
    int get(int id, int def) const;
    // get float
    // integer parameters are converted to float
    float get(int id, float def) const;
    // get array
    Mat get(int id, const Mat& def) const;
    // get string
    std::string get(int id, const std::string& def) const;

    // setters ignore invalid ids
    // set int
    void set(int id, int i);
    // set float
    void set(int id, float f);
    // set array
    void set(int id, const Mat& v);
    // set string
    void set(int id, const std::string& s);

protected:
    friend class Net;

    void clear();

    // failed loads may leave partially parsed parameters; discard them or load again
    int load_param(const DataReader& dr);
    int load_param_bin(const DataReader& dr);

private:
    ParamDictPrivate* const d;
};

} // namespace ncnn

#endif // NCNN_PARAMDICT_H
