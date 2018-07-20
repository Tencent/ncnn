// SenseNets is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2018 SenseNets Technology Ltd. All rights reserved.
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

#ifndef _QUANTIZE_H
#define _QUANTIZE_H

#include <iostream>
#include <string>
#include <sstream>  
#include <string>
#include <fstream>  
#include <stdlib.h>  
#include <vector>
#include "mat.h"

/*
 * Convert string type to need type
 */
template <class Type>
Type stringToNum(const std::string& str)
{
    std::istringstream iss(str);
    Type num;
    iss >> num;
    return num;
}

typedef struct _quantizeParams
{
    std::string name;
    float  dataScale;
    float  weightScale;
}stQuantizeParams;

typedef struct _quantizeParamsBin
{
    int index;
    float  dataScale;
    float  weightScale;
}stQuantizeParamsBin;

typedef enum _convModel
{
    CONV_FP32 = 0,
    CONV_INT8 = 1,
}enConvModel;

#endif //_QUANTIZE_H

