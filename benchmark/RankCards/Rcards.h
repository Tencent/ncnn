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
#ifndef RCARDS_H
#define RCARDS_H

#include <cstdint>
#include <cmath>
#include <deque>
#include <list>
#include <array>
#include <memory>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string.h>
#include <istream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <thread>

//---------------------------------------------------------------------------
// Global hardcoded parameters
//---------------------------------------------------------------------------
// LERP(a,b,c) = linear interpolation macro, is 'a' when c == 0.0 and 'b' when c == 1.0 */
#define MIN(a, b)                 ((a) > (b) ? (b) : (a))
#define MAX(a, b)                 ((a) < (b) ? (b) : (a))
#define LIM(a, b, c)              (((a) > (c)) ? (c) : ((a) < (b)) ? (b) : (a))
#define LERP(a, b, c)             (((b) - (a)) * (c) + (a))
#define ROUND(a)                  (static_cast<int>((a) + 0.5))
#define EUCLIDEAN(x1, y1, x2, y2) sqrt(((x1) - (x2)) * ((x1) - (x2)) + ((y1) - (y2)) * ((y1) - (y2)))
//---------------------------------------------------------------------------
struct TModel
{
    std::string Name;
    float AvrTime{0.0};
};
//---------------------------------------------------------------------------
struct TModelSet
{
    std::vector<TModel> Mset;

    //use push_back to prevent <brace-enclosed initializer list> issues with CMake
    inline TModelSet(void)
    {
        TModel model;
        model.Name = "squeezenet";
        Mset.push_back(model);
        model.Name = "squeezenet_int8";
        Mset.push_back(model);
        model.Name = "mobilenet";
        Mset.push_back(model);
        model.Name = "mobilenet_int8";
        Mset.push_back(model);
        model.Name = "mobilenet_v2";
        Mset.push_back(model);
        model.Name = "mobilenet_v3";
        Mset.push_back(model);
        model.Name = "shufflenet";
        Mset.push_back(model);
        model.Name = "shufflenet_v2";
        Mset.push_back(model);
        model.Name = "mnasnet";
        Mset.push_back(model);
        model.Name = "proxylessnasnet";
        Mset.push_back(model);
        model.Name = "efficientnet_b0";
        Mset.push_back(model);
        model.Name = "efficientnetv2_b0";
        Mset.push_back(model);
        model.Name = "regnety_400m";
        Mset.push_back(model);
        model.Name = "blazeface";
        Mset.push_back(model);
        model.Name = "googlenet";
        Mset.push_back(model);
        model.Name = "googlenet_int8";
        Mset.push_back(model);
        model.Name = "resnet18";
        Mset.push_back(model);
        model.Name = "resnet18_int8";
        Mset.push_back(model);
        model.Name = "alexnet";
        Mset.push_back(model);
        model.Name = "vgg16";
        Mset.push_back(model);
        model.Name = "vgg16_int8";
        Mset.push_back(model);
        model.Name = "resnet50";
        Mset.push_back(model);
        model.Name = "resnet50_int8";
        Mset.push_back(model);
        model.Name = "squeezenet_ssd";
        Mset.push_back(model);
        model.Name = "squeezenet_ssd_int8";
        Mset.push_back(model);
        model.Name = "mobilenet_ssd";
        Mset.push_back(model);
        model.Name = "mobilenet_ssd_int8";
        Mset.push_back(model);
        model.Name = "mobilenet_yolo";
        Mset.push_back(model);
        model.Name = "mobilenetv2_yolov3";
        Mset.push_back(model);
        model.Name = "yolov4-tiny";
        Mset.push_back(model);
        model.Name = "nanodet_m";
        Mset.push_back(model);
        model.Name = "yolo-fastest-1.1";
        Mset.push_back(model);
        model.Name = "yolo-fastestv2";
        Mset.push_back(model);
        model.Name = "vision_transformer";
        Mset.push_back(model);
        model.Name = "FastestDet";
        Mset.push_back(model);
    }

    void Store(const TModel& model)
    {
        for (size_t i = 0; i < Mset.size(); i++)
        {
            if (Mset[i].Name == model.Name)
            {
                Mset[i].AvrTime = model.AvrTime;
                break;
            }
        }
    }

    float Sum(void)
    {
        float t = 0;

        for (size_t i = 0; i < Mset.size(); i++) t += Mset[i].AvrTime;

        return t;
    }

    float Ratio(const TModelSet& Rset)
    {
        float w;
        float s = 0;
        float t = 0;

        for (size_t r = 0; r < Rset.Mset.size(); r++)
        {
            if (Rset.Mset[r].AvrTime > 0.0)
            {
                for (size_t i = 0; i < Mset.size(); i++)
                {
                    if (Mset[i].AvrTime > 0.0)
                    {
                        if (Mset[i].Name == Rset.Mset[r].Name)
                        {
                            w = log(Rset.Mset[r].AvrTime);
                            s += w * (Mset[i].AvrTime / Rset.Mset[r].AvrTime);
                            t += w;
                        }
                    }
                }
            }
        }
        if (t > 0) s /= t;
        return s;
    }
};
//---------------------------------------------------------------------------
struct TBoard
{
    std::string Name;
    size_t StartLine;
    size_t EndLine;
    std::vector<TModelSet> BenchSet;
    int BestSet;
    float Ratio;
};
//---------------------------------------------------------------------------
inline bool FileExists(const std::string& name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}
//---------------------------------------------------------------------------
inline void FileCopy(const std::string& Src, const std::string& Dst)
{
    std::ifstream src(Src, std::ios::binary);
    std::ofstream dst(Dst, std::ios::binary);

    dst << src.rdbuf();
}
//---------------------------------------------------------------------------
// to lower case
static inline void lcase(std::string& s)
{
    std::transform(s.begin(), s.end(), s.begin(),
    [](unsigned char c) {
        return std::tolower(c);
    });
}
//---------------------------------------------------------------------------
// to lower case (copying)
static inline std::string lcase_copy(std::string s)
{
    lcase(s);
    return s;
}
//---------------------------------------------------------------------------
// trim from start (in place)
static inline void ltrim(std::string& s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}
//---------------------------------------------------------------------------
// trim from end (in place)
static inline void rtrim(std::string& s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(),
    s.end());
}
//---------------------------------------------------------------------------
// trim from both ends (in place)
static inline void trim(std::string& s)
{
    ltrim(s);
    rtrim(s);
}
//---------------------------------------------------------------------------
// trim from start (copying)
static inline std::string ltrim_copy(std::string s)
{
    ltrim(s);
    return s;
}
//---------------------------------------------------------------------------
// trim from end (copying)
static inline std::string rtrim_copy(std::string s)
{
    rtrim(s);
    return s;
}
//---------------------------------------------------------------------------
// trim from both ends (copying)
static inline std::string trim_copy(std::string s)
{
    trim(s);
    return s;
}
//---------------------------------------------------------------------------
static inline void GetNameAver(std::string line, TModel& model)
{
    // line example: squeezenet  min =   46.28  max =   46.91  avg =   46.65

    size_t p = line.find("min =");

    if (p != std::string::npos)
    {
        model.Name = trim_copy(line.substr(0, p));
        p = line.find("avg =");
        if (p != std::string::npos)
        {
            try
            {
                model.AvrTime = std::stof(trim_copy(line.substr(p + 5, line.length() - p - 5)));
            }
            catch (...)
            {
            }
        }
        else
            model.AvrTime = 0.0;
    }
    else
    {
        model.Name = "";
        model.AvrTime = 0.0;
    }
}
//---------------------------------------------------------------------------
#endif // RCARDS_H
