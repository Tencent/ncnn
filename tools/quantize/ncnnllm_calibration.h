// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_LLM_CALIBRATION_H
#define NCNN_LLM_CALIBRATION_H

#include <algorithm>
#include <cmath>
#include <stdint.h>
#include <vector>

#include "mat.h"

class QuantActStat
{
public:
    QuantActStat()
    {
        width = 0;
        count = 0;
        max_sample_count = 0;
        random_state = 0;
        source_count = 0;
        last_source_id = -1;
    }

public:
    int width;
    uint64_t count;
    int max_sample_count;
    uint64_t random_state;
    int source_count;
    int last_source_id;
    ncnn::Mat sum_abs;
    std::vector<float> samples;
    std::vector<int> sample_source_ids;
};

static uint64_t next_sample_random(uint64_t& state)
{
    // SplitMix64 fixes the random stream independently of the standard library.
    state += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t v = state;
    v = (v ^ (v >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    v = (v ^ (v >> 27)) * UINT64_C(0x94d049bb133111eb);
    return v ^ (v >> 31);
}

static uint64_t sample_random_bounded(uint64_t& state, uint64_t bound)
{
    // Reject the incomplete residue range to avoid modulo bias.
    const uint64_t threshold = (uint64_t(0) - bound) % bound;
    uint64_t v;
    do
    {
        v = next_sample_random(state);
    } while (v < threshold);
    return v % bound;
}

static int init_act_stat(QuantActStat& stat, int width, int max_sample_count, uint64_t seed)
{
    if (width <= 0 || max_sample_count <= 0 || (size_t)max_sample_count > stat.samples.max_size() / (size_t)width || (size_t)max_sample_count > stat.sample_source_ids.max_size())
        return -100;

    stat.width = width;
    stat.count = 0;
    stat.max_sample_count = max_sample_count;
    stat.random_state = seed;
    stat.source_count = 0;
    stat.last_source_id = -1;

    stat.sum_abs.create(width);
    if (stat.sum_abs.empty())
        return -100;

    stat.sum_abs.fill(0.f);
    stat.samples.clear();
    stat.samples.reserve((size_t)width * max_sample_count);
    stat.sample_source_ids.clear();
    stat.sample_source_ids.reserve(max_sample_count);

    return 0;
}

static void collect_act_row(QuantActStat& stat, const float* ptr, int source_id)
{
    const int width = stat.width;
    float* sum_abs_ptr = stat.sum_abs;

    for (int k = 0; k < width; k++)
    {
        const float v = (float)fabs(ptr[k]);
        sum_abs_ptr[k] += v;
    }

    stat.count++;
    // Calibration entries are visited in order, so no unbounded source set is needed.
    if (stat.last_source_id != source_id)
    {
        stat.source_count++;
        stat.last_source_id = source_id;
    }

    if ((int)stat.sample_source_ids.size() < stat.max_sample_count)
    {
        stat.samples.insert(stat.samples.end(), ptr, ptr + width);
        stat.sample_source_ids.push_back(source_id);
        return;
    }

    const uint64_t j = sample_random_bounded(stat.random_state, stat.count);
    if (j < (uint64_t)stat.max_sample_count)
    {
        std::copy(ptr, ptr + width, stat.samples.begin() + (size_t)j * width);
        stat.sample_source_ids[(size_t)j] = source_id;
    }
}

static void collect_act_rows(QuantActStat& stat, const ncnn::Mat& m, int source_id)
{
    if (m.dims == 1)
    {
        collect_act_row(stat, (const float*)m, source_id);
        return;
    }

    if (m.dims == 2)
    {
        for (int y = 0; y < m.h; y++)
            collect_act_row(stat, m.row(y), source_id);
        return;
    }

    if (m.dims == 3)
    {
        for (int q = 0; q < m.c; q++)
        {
            const ncnn::Mat m0 = m.channel(q);
            for (int y = 0; y < m.h; y++)
            {
                collect_act_row(stat, m0.row(y), source_id);
            }
        }
        return;
    }

    if (m.dims == 4)
    {
        for (int q = 0; q < m.c; q++)
        {
            const ncnn::Mat m0 = m.channel(q);
            for (int z = 0; z < m.d; z++)
            {
                const ncnn::Mat m1 = m0.depth(z);
                for (int y = 0; y < m.h; y++)
                {
                    collect_act_row(stat, m1.row(y), source_id);
                }
            }
        }
        return;
    }
}

static int count_sample_sources(const QuantActStat& stat)
{
    std::vector<int> source_ids = stat.sample_source_ids;
    std::sort(source_ids.begin(), source_ids.end());
    return (int)(std::unique(source_ids.begin(), source_ids.end()) - source_ids.begin());
}

#endif // NCNN_LLM_CALIBRATION_H
