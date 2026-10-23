// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <limits.h>
#include <stdio.h>

#include "../tools/quantize/ncnnllm_calibration.h"

static int test_sample_count(int rows, int capacity)
{
    QuantActStat stat;
    if (init_act_stat(stat, 3, capacity, 7767517) != 0)
        return -1;

    const size_t sample_capacity = stat.samples.capacity();
    const size_t source_capacity = stat.sample_source_ids.capacity();
    for (int i = 0; i < rows; i++)
    {
        const float row[3] = {(float)i, (float)-i, (float)(i + 1)};
        collect_act_row(stat, row, i / 16);
    }

    const int expected_count = std::min(rows, capacity);
    const float* sum_abs = stat.sum_abs;
    if (stat.count != (uint64_t)rows || stat.samples.size() != (size_t)expected_count * 3 || stat.sample_source_ids.size() != (size_t)expected_count
            || sum_abs[0] != (float)(rows * (rows - 1) / 2) || sum_abs[1] != sum_abs[0] || sum_abs[2] != (float)(rows * (rows + 1) / 2)
            || stat.samples.capacity() != sample_capacity || stat.sample_source_ids.capacity() != source_capacity
            || stat.source_count != (rows + 15) / 16)
    {
        fprintf(stderr, "test_sample_count failed rows=%d capacity=%d\n", rows, capacity);
        return -1;
    }

    bool later_row = false;
    std::vector<int> selected(rows, 0);
    for (int i = 0; i < expected_count; i++)
    {
        const float* row = &stat.samples[(size_t)i * 3];
        const int row_id = (int)row[0];
        if (row_id < 0 || row_id >= rows || selected[row_id] || row[1] != -row[0] || row[2] != row[0] + 1
                || stat.sample_source_ids[i] != row_id / 16 || (rows <= capacity && row_id != i))
        {
            fprintf(stderr, "test_sample_count invalid row rows=%d capacity=%d slot=%d\n", rows, capacity, i);
            return -1;
        }
        selected[row_id] = 1;
        later_row = later_row || row_id >= capacity;
    }

    if (rows > capacity && !later_row)
    {
        fprintf(stderr, "test_sample_count retained only prefix rows=%d capacity=%d\n", rows, capacity);
        return -1;
    }

    std::vector<int> source_ids((rows + 15) / 16, 0);
    for (int i = 0; i < expected_count; i++)
        source_ids[stat.sample_source_ids[i]] = 1;
    int retained_sources = 0;
    for (size_t i = 0; i < source_ids.size(); i++)
        retained_sources += source_ids[i];
    if (count_sample_sources(stat) != retained_sources)
    {
        fprintf(stderr, "test_sample_count source coverage failed\n");
        return -1;
    }

    return 0;
}

static int test_sample_determinism()
{
    QuantActStat a;
    QuantActStat b;
    QuantActStat c;
    if (init_act_stat(a, 1, 8, 7767517) != 0 || init_act_stat(b, 1, 8, 7767517) != 0 || init_act_stat(c, 1, 8, 7767518) != 0)
        return -1;

    for (int i = 0; i < 64; i++)
    {
        const float v = (float)i;
        collect_act_row(a, &v, i / 16);
    }
    for (int i = 0; i < 64; i++)
    {
        const float v = (float)i;
        collect_act_row(c, &v, i / 16);
        collect_act_row(b, &v, i / 16);
    }

    if (a.samples != b.samples || a.sample_source_ids != b.sample_source_ids || a.random_state != b.random_state || a.samples == c.samples)
    {
        fprintf(stderr, "test_sample_determinism failed\n");
        return -1;
    }

    const float expected[8] = {15.f, 50.f, 2.f, 11.f, 36.f, 20.f, 62.f, 16.f};
    for (int i = 0; i < 8; i++)
    {
        if (a.samples[i] != expected[i])
        {
            fprintf(stderr, "test_sample_determinism selected rows changed slot=%d\n", i);
            return -1;
        }
    }

    uint64_t state = 0;
    if (next_sample_random(state) != UINT64_C(0xe220a8397b1dcdaf) || next_sample_random(state) != UINT64_C(0x6e789e6aa1b965f4))
    {
        fprintf(stderr, "test_sample_determinism random stream changed\n");
        return -1;
    }

    return 0;
}

static int test_sample_uniformity()
{
    const int trials = 4096;
    const int rows = 64;
    const int capacity = 8;
    int counts[rows] = {0};
    for (int seed = 0; seed < trials; seed++)
    {
        QuantActStat stat;
        if (init_act_stat(stat, 1, capacity, seed) != 0)
            return -1;
        for (int i = 0; i < rows; i++)
        {
            const float v = (float)i;
            collect_act_row(stat, &v, 0);
        }
        for (int i = 0; i < capacity; i++)
            counts[(int)stat.samples[i]]++;
    }

    const int expected = trials * capacity / rows;
    for (int i = 0; i < rows; i++)
    {
        if (counts[i] < expected - 100 || counts[i] > expected + 100)
        {
            fprintf(stderr, "test_sample_uniformity failed row=%d count=%d expected=%d\n", i, counts[i], expected);
            return -1;
        }
    }
    return 0;
}

static int test_sample_rows(int dims)
{
    ncnn::Mat m;
    if (dims == 1)
        m.create(3);
    if (dims == 2)
        m.create(3, 5);
    if (dims == 3)
        m.create(3, 5, 2);
    if (dims == 4)
        m.create(3, 5, 2, 2);
    if (m.empty())
        return -1;

    int rows = 0;
    for (int q = 0; q < m.c; q++)
    {
        ncnn::Mat channel = m.channel(q);
        for (int z = 0; z < m.d; z++)
        {
            ncnn::Mat depth = channel.depth(z);
            for (int y = 0; y < m.h; y++)
            {
                float* ptr = depth.row(y);
                for (int k = 0; k < m.w; k++)
                    ptr[k] = (float)(rows * 3 + k);
                rows++;
            }
        }
    }

    QuantActStat actual;
    QuantActStat expected;
    if (init_act_stat(actual, 3, 4, 7767517) != 0 || init_act_stat(expected, 3, 4, 7767517) != 0)
        return -1;
    collect_act_rows(actual, m, 7);
    for (int i = 0; i < rows; i++)
    {
        const float row[3] = {(float)(i * 3), (float)(i * 3 + 1), (float)(i * 3 + 2)};
        collect_act_row(expected, row, 7);
    }

    const float* actual_sum = actual.sum_abs;
    const float* expected_sum = expected.sum_abs;
    if (actual.count != expected.count || actual.samples != expected.samples || actual.sample_source_ids != expected.sample_source_ids
            || actual_sum[0] != expected_sum[0] || actual_sum[1] != expected_sum[1] || actual_sum[2] != expected_sum[2]
            || actual.source_count != 1 || count_sample_sources(actual) != 1)
    {
        fprintf(stderr, "test_sample_rows failed dims=%d\n", dims);
        return -1;
    }
    return 0;
}

static int test_sample_limits()
{
    QuantActStat stat;
    if (init_act_stat(stat, 0, 1, 0) == 0 || init_act_stat(stat, 1, 0, 0) == 0 || init_act_stat(stat, -1, 1, 0) == 0
            || init_act_stat(stat, 1, -1, 0) == 0 || init_act_stat(stat, INT_MAX, INT_MAX, 0) == 0)
    {
        fprintf(stderr, "test_sample_limits accepted invalid dimensions\n");
        return -1;
    }
    if (init_act_stat(stat, 1, 1, 0) != 0)
        return -1;

    float v = 1.f;
    collect_act_row(stat, &v, 0);
    stat.count = (uint64_t)INT_MAX;
    collect_act_row(stat, &v, 1);
    if (stat.count != (uint64_t)INT_MAX + 1 || stat.samples.size() != 1 || stat.source_count != 2 || count_sample_sources(stat) != 1)
    {
        fprintf(stderr, "test_sample_limits count overflow\n");
        return -1;
    }

    uint64_t state = 0;
    const uint64_t bounds[] = {1, 3, (uint64_t)INT_MAX + 1, UINT64_C(0x8000000000000001), UINT64_MAX};
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 100; j++)
        {
            if (sample_random_bounded(state, bounds[i]) >= bounds[i])
            {
                fprintf(stderr, "test_sample_limits random index out of range\n");
                return -1;
            }
        }
    }

    if (init_act_stat(stat, 1, 1, 0) != 0 || stat.count != 0 || stat.source_count != 0 || !stat.samples.empty() || !stat.sample_source_ids.empty() || count_sample_sources(stat) != 0)
    {
        fprintf(stderr, "test_sample_limits reset failed\n");
        return -1;
    }
    return 0;
}

int main()
{
    return 0
           || test_sample_count(0, 128)
           || test_sample_count(7, 128)
           || test_sample_count(128, 128)
           || test_sample_count(1024, 128)
           || test_sample_count(1024, 129)
           || test_sample_count(1024, 1)
           || test_sample_determinism()
           || test_sample_uniformity()
           || test_sample_rows(1)
           || test_sample_rows(2)
           || test_sample_rows(3)
           || test_sample_rows(4)
           || test_sample_limits();
}
