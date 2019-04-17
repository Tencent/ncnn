// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "cast.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Cast)

Cast::Cast()
{
    one_blob_only = true;
    support_inplace = false;
    support_vulkan = true;

#if NCNN_VULKAN
    pipeline_cast_fp32_to_fp16 = 0;
    pipeline_cast_fp32_to_fp16_pack4 = 0;
    pipeline_cast_fp16_to_fp32 = 0;
    pipeline_cast_fp16_to_fp32_pack4 = 0;
#endif // NCNN_VULKAN
}

int Cast::load_param(const ParamDict& pd)
{
    type_from = pd.get(0, 0);
    type_to = pd.get(1, 0);

    return 0;
}

// convert float to half precision floating point
static unsigned short float32_to_float16(float value)
{
    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;

    tmp.f = value;

    // 1 : 8 : 23
    unsigned short sign = (tmp.u & 0x80000000) >> 31;
    unsigned short exponent = (tmp.u & 0x7F800000) >> 23;
    unsigned int significand = tmp.u & 0x7FFFFF;

//     fprintf(stderr, "%d %d %d\n", sign, exponent, significand);

    // 1 : 5 : 10
    unsigned short fp16;
    if (exponent == 0)
    {
        // zero or denormal, always underflow
        fp16 = (sign << 15) | (0x00 << 10) | 0x00;
    }
    else if (exponent == 0xFF)
    {
        // infinity or NaN
        fp16 = (sign << 15) | (0x1F << 10) | (significand ? 0x200 : 0x00);
    }
    else
    {
        // normalized
        short newexp = exponent + (- 127 + 15);
        if (newexp >= 31)
        {
            // overflow, return infinity
            fp16 = (sign << 15) | (0x1F << 10) | 0x00;
        }
        else if (newexp <= 0)
        {
            // underflow
            if (newexp >= -10)
            {
                // denormal half-precision
                unsigned short sig = (significand | 0x800000) >> (14 - newexp);
                fp16 = (sign << 15) | (0x00 << 10) | sig;
            }
            else
            {
                // underflow
                fp16 = (sign << 15) | (0x00 << 10) | 0x00;
            }
        }
        else
        {
            fp16 = (sign << 15) | (newexp << 10) | (significand >> 13);
        }
    }

    return fp16;
}

// convert half precision floating point to float
static float float16_to_float32(unsigned short value)
{
    // 1 : 5 : 10
    unsigned short sign = (value & 0x8000) >> 15;
    unsigned short exponent = (value & 0x7c00) >> 10;
    unsigned short significand = value & 0x03FF;

//     fprintf(stderr, "%d %d %d\n", sign, exponent, significand);

    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;
    if (exponent == 0)
    {
        if (significand == 0)
        {
            // zero
            tmp.u = (sign << 31);
        }
        else
        {
            // denormal
            exponent = 0;
            // find non-zero bit
            while ((significand & 0x200) == 0)
            {
                significand <<= 1;
                exponent++;
            }
            significand <<= 1;
            significand &= 0x3FF;
            tmp.u = (sign << 31) | ((-exponent + (-15 + 127)) << 23) | (significand << 13);
        }
    }
    else if (exponent == 0x1F)
    {
        // infinity or NaN
        tmp.u = (sign << 31) | (0xFF << 23) | (significand << 13);
    }
    else
    {
        // normalized
        tmp.u = (sign << 31) | ((exponent + (-15 + 127)) << 23) | (significand << 13);
    }

    return tmp.f;
}

// round to nearest
static signed char float32_to_int8(float value)
{
    float tmp;
    if (value >= 0.f) tmp = value + 0.5;
    else tmp = value - 0.5;

    if (tmp > 127)
        return 127;
    if (tmp < -128)
        return -128;

    return tmp;
}

int Cast::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (type_from == type_to)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int packing = bottom_blob.packing;

    size_t out_elemsize = elemsize;
    if (type_to == 1)
    {
        // float32
        out_elemsize = 4 * packing;
    }
    else if (type_to == 2)
    {
        // float16
        out_elemsize = 2 * packing;
    }
    else if (type_to == 3)
    {
        // int8
        out_elemsize = packing;
    }

    if (dims == 1)
    {
        top_blob.create(w, out_elemsize, packing, opt.blob_allocator);
    }
    else if (dims == 2)
    {
        top_blob.create(w, h, out_elemsize, packing, opt.blob_allocator);
    }
    else if (dims == 3)
    {
        top_blob.create(w, h, channels, out_elemsize, packing, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    int size = w * h * packing;

    if (type_from == 1 && type_to == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            unsigned short* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = float32_to_float16(ptr[i]);
            }
        }
    }

    if (type_from == 2 && type_to == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = float16_to_float32(ptr[i]);
            }
        }
    }

    // TODO more cast type

    return 0;
}

#if NCNN_VULKAN
int Cast::create_pipeline()
{
    std::vector<vk_specialization_type> specializations;

    if (type_from == 1 && type_to == 2)
    {
        // pack1
        {
            pipeline_cast_fp32_to_fp16 = new Pipeline(vkdev);
            pipeline_cast_fp32_to_fp16->set_optimal_local_size_xyz();
            pipeline_cast_fp32_to_fp16->create("cast_fp32_to_fp16", specializations, 2, 10);
        }

        // pack4
        {
            pipeline_cast_fp32_to_fp16_pack4 = new Pipeline(vkdev);
            pipeline_cast_fp32_to_fp16_pack4->set_optimal_local_size_xyz();
            pipeline_cast_fp32_to_fp16_pack4->create("cast_fp32_to_fp16_pack4", specializations, 2, 10);
        }
    }

    if (type_from == 2 && type_to == 1)
    {
        // pack1
        {
            pipeline_cast_fp16_to_fp32 = new Pipeline(vkdev);
            pipeline_cast_fp16_to_fp32->set_optimal_local_size_xyz();
            pipeline_cast_fp16_to_fp32->create("cast_fp16_to_fp32", specializations, 2, 10);
        }

        // pack4
        {
            pipeline_cast_fp16_to_fp32_pack4 = new Pipeline(vkdev);
            pipeline_cast_fp16_to_fp32_pack4->set_optimal_local_size_xyz();
            pipeline_cast_fp16_to_fp32_pack4->create("cast_fp16_to_fp32_pack4", specializations, 2, 10);
        }
    }

    return 0;
}

int Cast::destroy_pipeline()
{
    delete pipeline_cast_fp32_to_fp16;
    pipeline_cast_fp32_to_fp16 = 0;

    delete pipeline_cast_fp32_to_fp16_pack4;
    pipeline_cast_fp32_to_fp16_pack4 = 0;

    delete pipeline_cast_fp16_to_fp32;
    pipeline_cast_fp16_to_fp32 = 0;

    delete pipeline_cast_fp16_to_fp32_pack4;
    pipeline_cast_fp16_to_fp32_pack4 = 0;

    return 0;
}

int Cast::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    if (type_from == type_to)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int packing = bottom_blob.packing;

    size_t out_elemsize = elemsize;
    if (type_to == 1)
    {
        // float32
        out_elemsize = 4 * packing;
    }
    else if (type_to == 2)
    {
        // float16
        out_elemsize = 2 * packing;
    }
    else if (type_to == 3)
    {
        // int8
        out_elemsize = packing;
    }

    if (dims == 1)
    {
        top_blob.create(w, out_elemsize, packing, opt.blob_vkallocator, opt.staging_vkallocator);
    }
    else if (dims == 2)
    {
        top_blob.create(w, h, out_elemsize, packing, opt.blob_vkallocator, opt.staging_vkallocator);
    }
    else if (dims == 3)
    {
        top_blob.create(w, h, channels, out_elemsize, packing, opt.blob_vkallocator, opt.staging_vkallocator);
    }
    if (top_blob.empty())
        return -100;

//     fprintf(stderr, "Cast::forward %p %p\n", bottom_blob.buffer(), top_blob.buffer());

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.c;
    constants[4].i = bottom_blob.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = top_blob.cstep;

    const Pipeline* pipeline = 0;

    if (type_from == 1 && type_to == 2)
    {
        pipeline = packing == 4 ? pipeline_cast_fp32_to_fp16_pack4 : pipeline_cast_fp32_to_fp16;
    }
    if (type_from == 2 && type_to == 1)
    {
        pipeline = packing == 4 ? pipeline_cast_fp16_to_fp32_pack4 : pipeline_cast_fp16_to_fp32;
    }

    // TODO more cast type

    // record
    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
